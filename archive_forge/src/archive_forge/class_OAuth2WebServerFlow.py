import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
class OAuth2WebServerFlow(Flow):
    """Does the Web Server Flow for OAuth 2.0.

    OAuth2WebServerFlow objects may be safely pickled and unpickled.
    """

    @_helpers.positional(4)
    def __init__(self, client_id, client_secret=None, scope=None, redirect_uri=None, user_agent=None, auth_uri=oauth2client.GOOGLE_AUTH_URI, token_uri=oauth2client.GOOGLE_TOKEN_URI, revoke_uri=oauth2client.GOOGLE_REVOKE_URI, login_hint=None, device_uri=oauth2client.GOOGLE_DEVICE_URI, token_info_uri=oauth2client.GOOGLE_TOKEN_INFO_URI, authorization_header=None, pkce=False, code_verifier=None, **kwargs):
        """Constructor for OAuth2WebServerFlow.

        The kwargs argument is used to set extra query parameters on the
        auth_uri. For example, the access_type and prompt
        query parameters can be set via kwargs.

        Args:
            client_id: string, client identifier.
            client_secret: string client secret.
            scope: string or iterable of strings, scope(s) of the credentials
                   being requested.
            redirect_uri: string, Either the string 'urn:ietf:wg:oauth:2.0:oob'
                          for a non-web-based application, or a URI that
                          handles the callback from the authorization server.
            user_agent: string, HTTP User-Agent to provide for this
                        application.
            auth_uri: string, URI for authorization endpoint. For convenience
                      defaults to Google's endpoints but any OAuth 2.0 provider
                      can be used.
            token_uri: string, URI for token endpoint. For convenience
                       defaults to Google's endpoints but any OAuth 2.0
                       provider can be used.
            revoke_uri: string, URI for revoke endpoint. For convenience
                        defaults to Google's endpoints but any OAuth 2.0
                        provider can be used.
            login_hint: string, Either an email address or domain. Passing this
                        hint will either pre-fill the email box on the sign-in
                        form or select the proper multi-login session, thereby
                        simplifying the login flow.
            device_uri: string, URI for device authorization endpoint. For
                        convenience defaults to Google's endpoints but any
                        OAuth 2.0 provider can be used.
            authorization_header: string, For use with OAuth 2.0 providers that
                                  require a client to authenticate using a
                                  header value instead of passing client_secret
                                  in the POST body.
            pkce: boolean, default: False, Generate and include a "Proof Key
                  for Code Exchange" (PKCE) with your authorization and token
                  requests. This adds security for installed applications that
                  cannot protect a client_secret. See RFC 7636 for details.
            code_verifier: bytestring or None, default: None, parameter passed
                           as part of the code exchange when pkce=True. If
                           None, a code_verifier will automatically be
                           generated as part of step1_get_authorize_url(). See
                           RFC 7636 for details.
            **kwargs: dict, The keyword arguments are all optional and required
                      parameters for the OAuth calls.
        """
        if scope is None:
            raise TypeError('The value of scope must not be None')
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = _helpers.scopes_to_string(scope)
        self.redirect_uri = redirect_uri
        self.login_hint = login_hint
        self.user_agent = user_agent
        self.auth_uri = auth_uri
        self.token_uri = token_uri
        self.revoke_uri = revoke_uri
        self.device_uri = device_uri
        self.token_info_uri = token_info_uri
        self.authorization_header = authorization_header
        self._pkce = pkce
        self.code_verifier = code_verifier
        self.params = _oauth2_web_server_flow_params(kwargs)

    @_helpers.positional(1)
    def step1_get_authorize_url(self, redirect_uri=None, state=None):
        """Returns a URI to redirect to the provider.

        Args:
            redirect_uri: string, Either the string 'urn:ietf:wg:oauth:2.0:oob'
                          for a non-web-based application, or a URI that
                          handles the callback from the authorization server.
                          This parameter is deprecated, please move to passing
                          the redirect_uri in via the constructor.
            state: string, Opaque state string which is passed through the
                   OAuth2 flow and returned to the client as a query parameter
                   in the callback.

        Returns:
            A URI as a string to redirect the user to begin the authorization
            flow.
        """
        if redirect_uri is not None:
            logger.warning('The redirect_uri parameter for OAuth2WebServerFlow.step1_get_authorize_url is deprecated. Please move to passing the redirect_uri in via the constructor.')
            self.redirect_uri = redirect_uri
        if self.redirect_uri is None:
            raise ValueError('The value of redirect_uri must not be None.')
        query_params = {'client_id': self.client_id, 'redirect_uri': self.redirect_uri, 'scope': self.scope}
        if state is not None:
            query_params['state'] = state
        if self.login_hint is not None:
            query_params['login_hint'] = self.login_hint
        if self._pkce:
            if not self.code_verifier:
                self.code_verifier = _pkce.code_verifier()
            challenge = _pkce.code_challenge(self.code_verifier)
            query_params['code_challenge'] = challenge
            query_params['code_challenge_method'] = 'S256'
        query_params.update(self.params)
        return _helpers.update_query_params(self.auth_uri, query_params)

    @_helpers.positional(1)
    def step1_get_device_and_user_codes(self, http=None):
        """Returns a user code and the verification URL where to enter it

        Returns:
            A user code as a string for the user to authorize the application
            An URL as a string where the user has to enter the code
        """
        if self.device_uri is None:
            raise ValueError('The value of device_uri must not be None.')
        body = urllib.parse.urlencode({'client_id': self.client_id, 'scope': self.scope})
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        if self.user_agent is not None:
            headers['user-agent'] = self.user_agent
        if http is None:
            http = transport.get_http_object()
        resp, content = transport.request(http, self.device_uri, method='POST', body=body, headers=headers)
        content = _helpers._from_bytes(content)
        if resp.status == http_client.OK:
            try:
                flow_info = json.loads(content)
            except ValueError as exc:
                raise OAuth2DeviceCodeError('Could not parse server response as JSON: "{0}", error: "{1}"'.format(content, exc))
            return DeviceFlowInfo.FromResponse(flow_info)
        else:
            error_msg = 'Invalid response {0}.'.format(resp.status)
            try:
                error_dict = json.loads(content)
                if 'error' in error_dict:
                    error_msg += ' Error: {0}'.format(error_dict['error'])
            except ValueError:
                pass
            raise OAuth2DeviceCodeError(error_msg)

    @_helpers.positional(2)
    def step2_exchange(self, code=None, http=None, device_flow_info=None):
        """Exchanges a code for OAuth2Credentials.

        Args:
            code: string, a dict-like object, or None. For a non-device
                  flow, this is either the response code as a string, or a
                  dictionary of query parameters to the redirect_uri. For a
                  device flow, this should be None.
            http: httplib2.Http, optional http instance to use when fetching
                  credentials.
            device_flow_info: DeviceFlowInfo, return value from step1 in the
                              case of a device flow.

        Returns:
            An OAuth2Credentials object that can be used to authorize requests.

        Raises:
            FlowExchangeError: if a problem occurred exchanging the code for a
                               refresh_token.
            ValueError: if code and device_flow_info are both provided or both
                        missing.
        """
        if code is None and device_flow_info is None:
            raise ValueError('No code or device_flow_info provided.')
        if code is not None and device_flow_info is not None:
            raise ValueError('Cannot provide both code and device_flow_info.')
        if code is None:
            code = device_flow_info.device_code
        elif not isinstance(code, (six.string_types, six.binary_type)):
            if 'code' not in code:
                raise FlowExchangeError(code.get('error', 'No code was supplied in the query parameters.'))
            code = code['code']
        post_data = {'client_id': self.client_id, 'code': code, 'scope': self.scope}
        if self.client_secret is not None:
            post_data['client_secret'] = self.client_secret
        if self._pkce:
            post_data['code_verifier'] = self.code_verifier
        if device_flow_info is not None:
            post_data['grant_type'] = 'http://oauth.net/grant_type/device/1.0'
        else:
            post_data['grant_type'] = 'authorization_code'
            post_data['redirect_uri'] = self.redirect_uri
        body = urllib.parse.urlencode(post_data)
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        if self.authorization_header is not None:
            headers['Authorization'] = self.authorization_header
        if self.user_agent is not None:
            headers['user-agent'] = self.user_agent
        if http is None:
            http = transport.get_http_object()
        resp, content = transport.request(http, self.token_uri, method='POST', body=body, headers=headers)
        d = _parse_exchange_token_response(content)
        if resp.status == http_client.OK and 'access_token' in d:
            access_token = d['access_token']
            refresh_token = d.get('refresh_token', None)
            if not refresh_token:
                logger.info("Received token response with no refresh_token. Consider reauthenticating with prompt='consent'.")
            token_expiry = None
            if 'expires_in' in d:
                delta = datetime.timedelta(seconds=int(d['expires_in']))
                token_expiry = delta + _UTCNOW()
            extracted_id_token = None
            id_token_jwt = None
            if 'id_token' in d:
                extracted_id_token = _extract_id_token(d['id_token'])
                id_token_jwt = d['id_token']
            logger.info('Successfully retrieved access token')
            return OAuth2Credentials(access_token, self.client_id, self.client_secret, refresh_token, token_expiry, self.token_uri, self.user_agent, revoke_uri=self.revoke_uri, id_token=extracted_id_token, id_token_jwt=id_token_jwt, token_response=d, scopes=self.scope, token_info_uri=self.token_info_uri)
        else:
            logger.info('Failed to retrieve access token: %s', content)
            if 'error' in d:
                error_msg = str(d['error']) + str(d.get('error_description', ''))
            else:
                error_msg = 'Invalid response: {0}.'.format(str(resp.status))
            raise FlowExchangeError(error_msg)