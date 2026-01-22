from __future__ import absolute_import
import io
import logging
import os
import random
import re
import time
import urllib
import httplib2
from oauth2client import client
from oauth2client import file as oauth2client_file
from oauth2client import tools
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine.tools.value_mixin import ValueMixin
from googlecloudsdk.third_party.appengine._internal import six_subset
class HttpRpcServerHttpLib2(object):
    """A variant of HttpRpcServer which uses httplib2.

  This follows the same interface as appengine_rpc.AbstractRpcServer,
  but is a totally separate implementation.
  """

    def __init__(self, host, auth_function, user_agent, source, host_override=None, extra_headers=None, save_cookies=False, auth_tries=None, account_type=None, debug_data=True, secure=True, ignore_certs=False, rpc_tries=3, conflict_max_errors=10, timeout_max_errors=2, http_class=None, http_object=None):
        """Creates a new HttpRpcServerHttpLib2.

    Args:
      host: The host to send requests to.
      auth_function: Saved but ignored; may be used by subclasses.
      user_agent: The user-agent string to send to the server. Specify None to
        omit the user-agent header.
      source: Saved but ignored; may be used by subclasses.
      host_override: The host header to send to the server (defaults to host).
      extra_headers: A dict of extra headers to append to every request. Values
        supplied here will override other default headers that are supplied.
      save_cookies: Saved but ignored; may be used by subclasses.
      auth_tries: The number of times to attempt auth_function before failing.
      account_type: Saved but ignored; may be used by subclasses.
      debug_data: Whether debugging output should include data contents.
      secure: If the requests sent using Send should be sent over HTTPS.
      ignore_certs: If the certificate mismatches should be ignored.
      rpc_tries: The number of rpc retries upon http server error (i.e.
        Response code >= 500 and < 600) before failing.
      conflict_max_errors: The number of rpc retries upon http server error
        (i.e. Response code 409) before failing.
      timeout_max_errors: The number of rpc retries upon http server timeout
        (i.e. Response code 408) before failing.
      http_class: the httplib2.Http subclass to use. Defaults to httplib2.Http.
      http_object: an httlib2.Http object to use to make requests. If this is
        provided, http_class is ignored.
    """
        self.host = host
        self.auth_function = auth_function
        self.user_agent = user_agent
        self.source = source
        self.host_override = host_override
        self.extra_headers = extra_headers or {}
        self.save_cookies = save_cookies
        self.auth_max_errors = auth_tries
        self.account_type = account_type
        self.debug_data = debug_data
        self.secure = secure
        self.ignore_certs = ignore_certs
        self.rpc_max_errors = rpc_tries
        self.scheme = secure and 'https' or 'http'
        self.conflict_max_errors = conflict_max_errors
        self.timeout_max_errors = timeout_max_errors
        self.http_class = http_class if http_class is not None else httplib2.Http
        self.http_object = http_object
        self.certpath = None
        self.cert_file_available = False
        if not self.ignore_certs:
            self.certpath = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'lib', 'cacerts', 'cacerts.txt'))
            self.cert_file_available = os.path.exists(self.certpath)
        self.memory_cache = MemoryCache()

    def _Authenticate(self, http, saw_error):
        """Pre or Re-auth stuff...

    Args:
      http: An 'Http' object from httplib2.
      saw_error: If the user has already tried to contact the server.
        If they have, it's OK to prompt them. If not, we should not be asking
        them for auth info--it's possible it'll suceed w/o auth.
    """
        raise NotImplementedError()

    def Send(self, request_path, payload='', content_type='application/octet-stream', timeout=None, **kwargs):
        """Sends an RPC and returns the response.

    Args:
      request_path: The path to send the request to, eg /api/appversion/create.
      payload: The body of the request, or None to send an empty request.
      content_type: The Content-Type header to use.
      timeout: timeout in seconds; default None i.e. no timeout.
        (Note: for large requests on OS X, the timeout doesn't work right.)
      Any keyword arguments are converted into query string parameters.

    Returns:
      The response body, as a string.

    Raises:
      AuthPermanentFail: If authorization failed in a permanent way.
      urllib2.HTTPError: On most HTTP errors.
    """
        self.http = self.http_object or self.http_class(cache=self.memory_cache, ca_certs=self.certpath, disable_ssl_certificate_validation=not self.cert_file_available)
        self.http.follow_redirects = False
        self.http.timeout = timeout
        url = '%s://%s%s' % (self.scheme, self.host, request_path)
        if kwargs:
            url += '?' + urlencode_fn(sorted(kwargs.items()))
        headers = {}
        if self.extra_headers:
            headers.update(self.extra_headers)
        headers['X-appcfg-api-version'] = '1'
        if payload is not None:
            method = 'POST'
            headers['content-length'] = str(len(payload))
            headers['Content-Type'] = content_type
        else:
            method = 'GET'
        if self.host_override:
            headers['Host'] = self.host_override
        rpc_errors = 0
        auth_errors = [0]
        conflict_errors = 0
        timeout_errors = 0

        def NeedAuth():
            """Marker that we need auth; it'll actually be tried next time around."""
            auth_errors[0] += 1
            logger.debug('Attempting to auth. This is try %s of %s.', auth_errors[0], self.auth_max_errors)
            if auth_errors[0] > self.auth_max_errors:
                RaiseHttpError(url, response_info, response, 'Too many auth attempts.')
        while rpc_errors < self.rpc_max_errors and conflict_errors < self.conflict_max_errors and (timeout_errors < self.timeout_max_errors):
            self._Authenticate(self.http, auth_errors[0] > 0)
            logger.debug('Sending request to %s headers=%s body=%s', url, headers, self.debug_data and payload or (payload and 'ELIDED') or '')
            try:
                response_info, response = self.http.request(url, method=method, body=payload, headers=headers)
            except client.AccessTokenRefreshError as e:
                logger.info('Got access token error', exc_info=1)
                response_info = httplib2.Response({'status': 401})
                response_info.reason = str(e)
                response = ''
            status = response_info.status
            if status == 200:
                return response
            logger.debug('Got http error %s.', response_info.status)
            if status == 401:
                NeedAuth()
                continue
            elif status == 408:
                timeout_errors += 1
                logger.debug('Got timeout error %s of %s. Retrying in %s seconds', timeout_errors, self.timeout_max_errors, _TIMEOUT_WAIT_TIME)
                time.sleep(_TIMEOUT_WAIT_TIME)
                continue
            elif status == 409:
                conflict_errors += 1
                wait_time = random.randint(0, 10)
                logger.debug('Got conflict error %s of %s. Retrying in %s seconds.', conflict_errors, self.conflict_max_errors, wait_time)
                time.sleep(wait_time)
                continue
            elif status >= 500 and status < 600:
                rpc_errors += 1
                logger.debug('Retrying. This is attempt %s of %s.', rpc_errors, self.rpc_max_errors)
                continue
            elif status == 302:
                loc = response_info.get('location')
                logger.debug('Got 302 redirect. Location: %s', loc)
                if loc.startswith('https://www.google.com/accounts/ServiceLogin') or re.match('https://www\\.google\\.com/a/[a-z0-9.-]+/ServiceLogin', loc):
                    NeedAuth()
                    continue
                elif loc.startswith('http://%s/_ah/login' % (self.host,)):
                    RaiseHttpError(url, response_info, response, 'dev_appserver login not supported')
                else:
                    RaiseHttpError(url, response_info, response, 'Unexpected redirect to %s' % loc)
            else:
                logger.debug('Unexpected results: %s', response_info)
                RaiseHttpError(url, response_info, response, 'Unexpected HTTP status %s' % status)
        logging.info('Too many retries for url %s', url)
        RaiseHttpError(url, response_info, response)