import abc
import base64
import hashlib
import os
import time
from urllib import parse as urlparse
import warnings
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import federation
class OidcDeviceAuthorization(_OidcBase):
    """Implementation for OAuth 2.0 Device Authorization Grant."""
    grant_type = 'urn:ietf:params:oauth:grant-type:device_code'
    HEADER_X_FORM = {'Content-Type': 'application/x-www-form-urlencoded'}

    def __init__(self, auth_url, identity_provider, protocol, client_id, client_secret=None, access_token_endpoint=None, device_authorization_endpoint=None, discovery_endpoint=None, code_challenge=None, code_challenge_method=None, **kwargs):
        """The OAuth 2.0 Device Authorization plugin expects the following.

        :param device_authorization_endpoint: OAuth 2.0 Device Authorization
                                  Endpoint, for example:
                                  https://localhost:8020/oidc/authorize/device
                                  Note that if a discovery document is
                                  provided this value will override
                                  the discovered one.
        :type device_authorization_endpoint: string

        :param code_challenge_method: PKCE Challenge Method (RFC 7636).
        :type code_challenge_method: string
        """
        self.access_token_type = 'access_token'
        self.device_authorization_endpoint = device_authorization_endpoint
        self.code_challenge_method = code_challenge_method
        super(OidcDeviceAuthorization, self).__init__(auth_url=auth_url, identity_provider=identity_provider, protocol=protocol, client_id=client_id, client_secret=client_secret, access_token_endpoint=access_token_endpoint, discovery_endpoint=discovery_endpoint, access_token_type=self.access_token_type, **kwargs)

    def _get_device_authorization_endpoint(self, session):
        """Get the endpoint for the OAuth 2.0 Device Authorization flow.

        This method will return the correct device authorization endpoint to
        be used.
        If the user has explicitly passed an device_authorization_endpoint to
        the constructor that will be returned. If there is no explicit endpoint
        and a discovery url is provided, it will try to get it from the
        discovery document. If nothing is found, an exception will be raised.

        :param session: a session object to send out HTTP requests.
        :type session: keystoneauth1.session.Session

        :return: the endpoint to use
        :rtype: string or None if no endpoint is found
        """
        if self.device_authorization_endpoint is not None:
            return self.device_authorization_endpoint
        discovery = self._get_discovery_document(session)
        endpoint = discovery.get('device_authorization_endpoint')
        if endpoint is None:
            raise exceptions.oidc.OidcDeviceAuthorizationEndpointNotFound()
        return endpoint

    def _generate_pkce_verifier(self):
        """Generate PKCE verifier string as defined in RFC 7636."""
        raw_bytes = 42
        _rand = os.urandom(raw_bytes)
        _rand_b64 = base64.urlsafe_b64encode(_rand).decode('ascii')
        code_verifier = _rand_b64.rstrip('=')
        return code_verifier

    def _generate_pkce_challenge(self):
        """Generate PKCE challenge string as defined in RFC 7636."""
        if self.code_challenge_method not in ('plain', 'S256'):
            raise exceptions.OidcGrantTypeMissmatch()
        self.code_verifier = self._generate_pkce_verifier()
        if self.code_challenge_method == 'plain':
            return self.code_verifier
        elif self.code_challenge_method == 'S256':
            _tmp = self.code_verifier.encode('ascii')
            _hash = hashlib.sha256(_tmp).digest()
            _tmp = base64.urlsafe_b64encode(_hash).decode('ascii')
            code_challenge = _tmp.rstrip('=')
            return code_challenge

    def get_payload(self, session):
        """Get an authorization grant for the "device_code" grant type.

        :param session: a session object to send out HTTP requests.
        :type session: keystoneauth1.session.Session

        :returns: a python dictionary containing the payload to be exchanged
        :rtype: dict
        """
        client_auth = (self.client_id, self.client_secret)
        device_authz_endpoint = self._get_device_authorization_endpoint(session)
        payload = {}
        if self.code_challenge_method:
            self.code_challenge = self._generate_pkce_challenge()
            payload.setdefault('code_challenge_method', self.code_challenge_method)
            payload.setdefault('code_challenge', self.code_challenge)
        encoded_payload = urlparse.urlencode(payload)
        op_response = session.post(device_authz_endpoint, requests_auth=client_auth, headers=self.HEADER_X_FORM, data=encoded_payload, authenticated=False)
        self.expires_in = int(op_response.json()['expires_in'])
        self.timeout = time.time() + self.expires_in
        self.device_code = op_response.json()['device_code']
        self.interval = int(op_response.json()['interval'])
        self.user_code = op_response.json()['user_code']
        self.verification_uri = op_response.json()['verification_uri']
        self.verification_uri_complete = op_response.json()['verification_uri_complete']
        payload = {'device_code': self.device_code}
        if self.code_challenge_method:
            payload.setdefault('code_verifier', self.code_verifier)
        return payload

    def _get_access_token(self, session, payload):
        """Poll token endpoint for an access token.

        :param session: a session object to send out HTTP requests.
        :type session: keystoneauth1.session.Session

        :param payload: a dict containing various OpenID Connect values,
                for example::
                {'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
                 'device_code': self.device_code}
        :type payload: dict
        """
        print(f'\nTo authenticate please go to: {self.verification_uri_complete}')
        client_auth = (self.client_id, self.client_secret)
        access_token_endpoint = self._get_access_token_endpoint(session)
        encoded_payload = urlparse.urlencode(payload)
        while time.time() < self.timeout:
            try:
                op_response = session.post(access_token_endpoint, requests_auth=client_auth, data=encoded_payload, headers=self.HEADER_X_FORM, authenticated=False)
            except exceptions.http.BadRequest as exc:
                error = exc.response.json().get('error')
                if error != 'authorization_pending':
                    raise
                time.sleep(self.interval)
                continue
            break
        else:
            if error == 'authorization_pending':
                raise exceptions.oidc.OidcDeviceAuthorizationTimeOut()
        access_token = op_response.json()[self.access_token_type]
        return access_token