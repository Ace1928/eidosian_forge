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
class HttpRpcServerOAuth2(HttpRpcServerHttpLib2):
    """A variant of HttpRpcServer which uses oauth2.

  This variant is specifically meant for interactive command line usage,
  as it will attempt to open a browser and ask the user to enter
  information from the resulting web page.
  """

    class OAuth2Parameters(ValueMixin):
        """Class encapsulating parameters related to OAuth2 authentication."""

        def __init__(self, access_token, client_id, client_secret, scope, refresh_token, credential_file, token_uri=None, credentials=None):
            self.access_token = access_token
            self.client_id = client_id
            self.client_secret = client_secret
            self.scope = scope
            self.refresh_token = refresh_token
            self.credential_file = credential_file
            self.token_uri = token_uri
            self.credentials = credentials

    class FlowFlags(object):

        def __init__(self, options):
            self.logging_level = logging.getLevelName(logging.getLogger().level)
            self.noauth_local_webserver = not options.auth_local_webserver if options else True
            self.auth_host_port = [8080, 8090]
            self.auth_host_name = 'localhost'

    def __init__(self, host, oauth2_parameters, user_agent, source, host_override=None, extra_headers=None, save_cookies=False, auth_tries=None, account_type=None, debug_data=True, secure=True, ignore_certs=False, rpc_tries=3, timeout_max_errors=2, options=None, http_class=None, http_object=None):
        """Creates a new HttpRpcServerOAuth2.

    Args:
      host: The host to send requests to.
      oauth2_parameters: An object of type OAuth2Parameters (defined above)
        that specifies all parameters related to OAuth2 authentication. (This
        replaces the auth_function parameter in the parent class.)
      user_agent: The user-agent string to send to the server. Specify None to
        omit the user-agent header.
      source: Saved but ignored.
      host_override: The host header to send to the server (defaults to host).
      extra_headers: A dict of extra headers to append to every request. Values
        supplied here will override other default headers that are supplied.
      save_cookies: If the refresh token should be saved.
      auth_tries: The number of times to attempt auth_function before failing.
      account_type: Ignored.
      debug_data: Whether debugging output should include data contents.
      secure: If the requests sent using Send should be sent over HTTPS.
      ignore_certs: If the certificate mismatches should be ignored.
      rpc_tries: The number of rpc retries upon http server error (i.e.
        Response code >= 500 and < 600) before failing.
      timeout_max_errors: The number of rpc retries upon http server timeout
        (i.e. Response code 408) before failing.
      options: the command line options.
      http_class: the httplib2.Http subclass to use. Defaults to httplib2.Http.
      http_object: an httlib2.Http object to use to make requests. If this is
        provided, http_class is ignored.
    """
        super(HttpRpcServerOAuth2, self).__init__(host, None, user_agent, source, host_override=host_override, extra_headers=extra_headers, auth_tries=auth_tries, debug_data=debug_data, secure=secure, ignore_certs=ignore_certs, rpc_tries=rpc_tries, timeout_max_errors=timeout_max_errors, save_cookies=save_cookies, http_class=http_class, http_object=http_object)
        if not isinstance(oauth2_parameters, self.OAuth2Parameters):
            raise TypeError('oauth2_parameters must be an OAuth2Parameters: %r' % oauth2_parameters)
        self.oauth2_parameters = oauth2_parameters
        if save_cookies:
            oauth2_credential_file = oauth2_parameters.credential_file or '~/.appcfg_oauth2_tokens'
            self.storage = oauth2client_file.Storage(os.path.expanduser(oauth2_credential_file))
        else:
            self.storage = NoStorage()
        if oauth2_parameters.credentials:
            self.credentials = oauth2_parameters.credentials
        elif any((oauth2_parameters.access_token, oauth2_parameters.refresh_token, oauth2_parameters.token_uri)):
            token_uri = oauth2_parameters.token_uri or 'https://%s/o/oauth2/token' % encoding.GetEncodedValue(os.environ, 'APPENGINE_AUTH_SERVER', 'accounts.google.com')
            self.credentials = client.OAuth2Credentials(oauth2_parameters.access_token, oauth2_parameters.client_id, oauth2_parameters.client_secret, oauth2_parameters.refresh_token, None, token_uri, self.user_agent)
        else:
            self.credentials = self.storage.get()
        self.flags = self.FlowFlags(options)

    def _Authenticate(self, http, needs_auth):
        """Pre or Re-auth stuff...

    This will attempt to avoid making any OAuth related HTTP connections or
    user interactions unless it's needed.

    Args:
      http: An 'Http' object from httplib2.
      needs_auth: If the user has already tried to contact the server.
        If they have, it's OK to prompt them. If not, we should not be asking
        them for auth info--it's possible it'll suceed w/o auth, but if we have
        some credentials we'll use them anyway.

    Raises:
      AuthPermanentFail: The user has requested non-interactive auth but
        the token is invalid.
    """
        if needs_auth and (not self.credentials or self.credentials.invalid):
            if self.oauth2_parameters.access_token:
                logger.debug('_Authenticate skipping auth because user explicitly supplied an access token.')
                raise AuthPermanentFail('Access token is invalid.')
            if self.oauth2_parameters.refresh_token:
                logger.debug('_Authenticate skipping auth because user explicitly supplied a refresh token.')
                raise AuthPermanentFail('Refresh token is invalid.')
            if self.oauth2_parameters.token_uri:
                logger.debug('_Authenticate skipping auth because user explicitly supplied a Token URI, for example for service account authentication with Compute Engine')
                raise AuthPermanentFail('Token URI did not yield a valid token: ' + self.oauth_parameters.token_uri)
            logger.debug('_Authenticate requesting auth')
            flow = client.OAuth2WebServerFlow(client_id=self.oauth2_parameters.client_id, client_secret=self.oauth2_parameters.client_secret, scope=_ScopesToString(self.oauth2_parameters.scope), user_agent=self.user_agent)
            self.credentials = tools.run_flow(flow, self.storage, self.flags)
        if self.credentials and (not self.credentials.invalid):
            if not self.credentials.access_token_expired or needs_auth:
                logger.debug('_Authenticate configuring auth; needs_auth=%s', needs_auth)
                self.credentials.authorize(http)
                return
        logger.debug('_Authenticate skipped auth; needs_auth=%s', needs_auth)