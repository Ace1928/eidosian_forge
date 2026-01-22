import gzip
import hashlib
import io
import logging
import os
import re
import socket
import sys
import time
import urllib
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine._internal import six_subset
class AbstractRpcServer(object):
    """Provides a common interface for a simple RPC server."""
    RUNTIME = 'python'

    def __init__(self, host, auth_function, user_agent, source, host_override=None, extra_headers=None, save_cookies=False, auth_tries=3, account_type=None, debug_data=True, secure=True, ignore_certs=False, rpc_tries=3, options=None):
        """Creates a new HttpRpcServer.

    Args:
      host: The host to send requests to.
      auth_function: A function that takes no arguments and returns an
        (email, password) tuple when called. Will be called if authentication
        is required.
      user_agent: The user-agent string to send to the server. Specify None to
        omit the user-agent header.
      source: The source to specify in authentication requests.
      host_override: The host header to send to the server (defaults to host).
      extra_headers: A dict of extra headers to append to every request. Values
        supplied here will override other default headers that are supplied.
      save_cookies: If True, save the authentication cookies to local disk.
        If False, use an in-memory cookiejar instead.  Subclasses must
        implement this functionality.  Defaults to False.
      auth_tries: The number of times to attempt auth_function before failing.
      account_type: One of GOOGLE, HOSTED_OR_GOOGLE, or None for automatic.
      debug_data: Whether debugging output should include data contents.
      secure: If the requests sent using Send should be sent over HTTPS.
      ignore_certs: If the certificate mismatches should be ignored.
      rpc_tries: The number of rpc retries upon http server error (i.e.
        Response code >= 500 and < 600) before failing.
      options: the command line options (ignored in this implementation).
    """
        if secure:
            self.scheme = 'https'
        else:
            self.scheme = 'http'
        self.ignore_certs = ignore_certs
        self.host = host
        self.host_override = host_override
        self.auth_function = auth_function
        self.source = source
        self.authenticated = False
        self.auth_tries = auth_tries
        self.debug_data = debug_data
        self.rpc_tries = rpc_tries
        self.account_type = account_type
        self.extra_headers = {}
        if user_agent:
            self.extra_headers['User-Agent'] = user_agent
        if extra_headers:
            self.extra_headers.update(extra_headers)
        self.save_cookies = save_cookies
        self.cookie_jar = MozillaCookieJar()
        self.opener = self._GetOpener()
        if self.host_override:
            logger.debug('Server: %s; Host: %s', self.host, self.host_override)
        else:
            logger.debug('Server: %s', self.host)
        if self.host_override and self.host_override == 'localhost' or self.host == 'localhost' or self.host.startswith('localhost:'):
            self._DevAppServerAuthenticate()

    def _GetOpener(self):
        """Returns an OpenerDirector for making HTTP requests.

    Returns:
      A urllib2.OpenerDirector object.
    """
        raise NotImplementedError

    def _CreateRequest(self, url, data=None):
        """Creates a new urllib request."""
        req = Request(url, data=data)
        if self.host_override:
            req.add_header('Host', self.host_override)
        for key, value in self.extra_headers.items():
            req.add_header(key, value)
        return req

    def _GetAuthToken(self, email, password):
        """Uses ClientLogin to authenticate the user, returning an auth token.

    Args:
      email:    The user's email address
      password: The user's password

    Raises:
      ClientLoginError: If there was an error authenticating with ClientLogin.
      HTTPError: If there was some other form of HTTP error.

    Returns:
      The authentication token returned by ClientLogin.
    """
        account_type = self.account_type
        if not account_type:
            if self.host.split(':')[0].endswith('.google.com') or (self.host_override and self.host_override.split(':')[0].endswith('.google.com')):
                account_type = 'HOSTED_OR_GOOGLE'
            else:
                account_type = 'GOOGLE'
        data = {'Email': email, 'Passwd': password, 'service': 'ah', 'source': self.source, 'accountType': account_type}
        req = self._CreateRequest(url='https://%s/accounts/ClientLogin' % encoding.GetEncodedValue(os.environ, 'APPENGINE_AUTH_SERVER', 'www.google.com'), data=urlencode_fn(data))
        try:
            response = self.opener.open(req)
            response_body = response.read()
            response_dict = dict((x.split('=') for x in response_body.split('\n') if x))
            if encoding.GetEncodedValue(os.environ, 'APPENGINE_RPC_USE_SID', '0') == '1':
                self.extra_headers['Cookie'] = 'SID=%s; Path=/;' % response_dict['SID']
            return response_dict['Auth']
        except HTTPError as e:
            if e.code == 403:
                body = e.read()
                response_dict = dict((x.split('=', 1) for x in body.split('\n') if x))
                raise ClientLoginError(req.get_full_url(), e.code, e.msg, e.headers, response_dict)
            else:
                raise

    def _GetAuthCookie(self, auth_token):
        """Fetches authentication cookies for an authentication token.

    Args:
      auth_token: The authentication token returned by ClientLogin.

    Raises:
      HTTPError: If there was an error fetching the authentication cookies.
    """
        continue_location = 'http://localhost/'
        args = {'continue': continue_location, 'auth': auth_token}
        login_path = os.environ.get('APPCFG_LOGIN_PATH', '/_ah')
        req = self._CreateRequest('%s://%s%s/login?%s' % (self.scheme, self.host, login_path, urlencode_fn(args)))
        try:
            response = self.opener.open(req)
        except HTTPError as e:
            response = e
        if response.code != 302 or response.info()['location'] != continue_location:
            raise HTTPError(req.get_full_url(), response.code, response.msg, response.headers, response.fp)
        self.authenticated = True

    def _Authenticate(self):
        """Authenticates the user.

    The authentication process works as follows:
     1) We get a username and password from the user
     2) We use ClientLogin to obtain an AUTH token for the user
        (see http://code.google.com/apis/accounts/AuthForInstalledApps.html).
     3) We pass the auth token to /_ah/login on the server to obtain an
        authentication cookie. If login was successful, it tries to redirect
        us to the URL we provided.

    If we attempt to access the upload API without first obtaining an
    authentication cookie, it returns a 401 response and directs us to
    authenticate ourselves with ClientLogin.
    """
        for unused_i in range(self.auth_tries):
            credentials = self.auth_function()
            try:
                auth_token = self._GetAuthToken(credentials[0], credentials[1])
                if encoding.GetEncodedValue(os.environ, 'APPENGINE_RPC_USE_SID', '0') == '1':
                    return
            except ClientLoginError as e:
                if e.reason == 'CaptchaRequired':
                    (print >> sys.stderr, 'Please go to\nhttps://www.google.com/accounts/DisplayUnlockCaptcha\nand verify you are a human.  Then try again.')
                    break
                if e.reason == 'NotVerified':
                    (print >> sys.stderr, 'Account not verified.')
                    break
                if e.reason == 'TermsNotAgreed':
                    (print >> sys.stderr, 'User has not agreed to TOS.')
                    break
                if e.reason == 'AccountDeleted':
                    (print >> sys.stderr, 'The user account has been deleted.')
                    break
                if e.reason == 'AccountDisabled':
                    (print >> sys.stderr, 'The user account has been disabled.')
                    break
                if e.reason == 'ServiceDisabled':
                    (print >> sys.stderr, "The user's access to the service has been disabled.")
                    break
                if e.reason == 'ServiceUnavailable':
                    (print >> sys.stderr, 'The service is not available; try again later.')
                    break
                raise
            self._GetAuthCookie(auth_token)
            return

    @staticmethod
    def _CreateDevAppServerCookieData(email, admin):
        """Creates cookie payload data.

    Args:
      email: The user's email address.
      admin: True if the user is an admin; False otherwise.

    Returns:
      String containing the cookie payload.
    """
        if email:
            user_id_digest = hashlib.md5(email.lower()).digest()
            user_id = '1' + ''.join(['%02d' % x for x in six_subset.iterbytes(user_id_digest)])[:20]
        else:
            user_id = ''
        return '%s:%s:%s' % (email, bool(admin), user_id)

    def _DevAppServerAuthenticate(self):
        """Authenticates the user on the dev_appserver."""
        credentials = self.auth_function()
        value = self._CreateDevAppServerCookieData(credentials[0], True)
        self.extra_headers['Cookie'] = 'dev_appserver_login="%s"; Path=/;' % value

    def Send(self, request_path, payload='', content_type='application/octet-stream', timeout=None, **kwargs):
        """Sends an RPC and returns the response.

    Args:
      request_path: The path to send the request to, eg /api/appversion/create.
      payload: The body of the request, or None to send an empty request.
      content_type: The Content-Type header to use.
      timeout: timeout in seconds; default None i.e. no timeout.
        (Note: for large requests on OS X, the timeout doesn't work right.)
      kwargs: Any keyword arguments are converted into query string parameters.

    Returns:
      The response body, as a string.
    """
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        try:
            tries = 0
            auth_tried = False
            while True:
                tries += 1
                url = '%s://%s%s' % (self.scheme, self.host, request_path)
                if kwargs:
                    url += '?' + urlencode_fn(sorted(kwargs.items()))
                req = self._CreateRequest(url=url, data=payload)
                req.add_header('Content-Type', content_type)
                req.add_header('X-appcfg-api-version', '1')
                try:
                    logger.debug('Sending %s request:\n%s', self.scheme.upper(), HttpRequestToString(req, include_data=self.debug_data))
                    f = self.opener.open(req)
                    response = f.read()
                    f.close()
                    return response
                except HTTPError as e:
                    logger.debug('Got http error, this is try %d: %s', tries, e)
                    if tries > self.rpc_tries:
                        raise
                    elif e.code == 401:
                        if auth_tried:
                            raise
                        auth_tried = True
                        self._Authenticate()
                    elif e.code >= 500 and e.code < 600:
                        continue
                    elif e.code == 302:
                        if auth_tried:
                            raise
                        auth_tried = True
                        loc = e.info()['location']
                        logger.debug('Got 302 redirect. Location: %s', loc)
                        if loc.startswith('https://www.google.com/accounts/ServiceLogin'):
                            self._Authenticate()
                        elif re.match('https://www\\.google\\.com/a/[a-z0-9\\.\\-]+/ServiceLogin', loc):
                            self.account_type = encoding.GetEncodedValue(os.environ, 'APPENGINE_RPC_HOSTED_LOGIN_TYPE', 'HOSTED')
                            self._Authenticate()
                        elif loc.startswith('http://%s/_ah/login' % (self.host,)):
                            self._DevAppServerAuthenticate()
                        else:
                            raise
                    else:
                        raise
        finally:
            socket.setdefaulttimeout(old_timeout)