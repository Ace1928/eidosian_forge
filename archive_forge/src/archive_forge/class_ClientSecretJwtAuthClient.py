import abc
import copy
import hashlib
import os
import ssl
import time
import uuid
import jwt.utils
import oslo_cache
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
import requests.auth
import webob.dec
import webob.exc
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from keystoneauth1.loading import session as session_loading
from keystonemiddleware._common import config
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.exceptions import ConfigurationError
from keystonemiddleware.exceptions import KeystoneMiddlewareException
from keystonemiddleware.i18n import _
class ClientSecretJwtAuthClient(AbstractAuthClient):
    """Http client with the auth method 'client_secret_jwt'."""

    def __init__(self, session, introspect_endpoint, audience, client_id, func_get_config_option, logger):
        super(ClientSecretJwtAuthClient, self).__init__(session, introspect_endpoint, audience, client_id, func_get_config_option, logger)
        self.client_secret = self.get_config_option('client_secret', is_required=True)
        self.jwt_bearer_time_out = self.get_config_option('jwt_bearer_time_out', is_required=True)
        self.jwt_algorithm = self.get_config_option('jwt_algorithm', is_required=True)

    def introspect(self, access_token):
        """Access the introspect API.

        Access the Introspect API to verify the access token by
        the auth method 'client_secret_jwt'.
        """
        ita = round(time.time())
        try:
            client_assertion = jwt.encode(payload={'jti': str(uuid.uuid4()), 'iat': str(ita), 'exp': str(ita + self.jwt_bearer_time_out), 'iss': self.client_id, 'sub': self.client_id, 'aud': self.audience}, headers={'typ': 'JWT', 'alg': self.jwt_algorithm}, key=self.client_secret, algorithm=self.jwt_algorithm)
        except Exception as e:
            self.logger.critical('Configuration error. JWT encoding with the specified client_secret and algorithm failed. algorithm: %s, error: %s' % (self.jwt_algorithm, e))
            raise ConfigurationError(_('Configuration error. JWT encoding with the specified client_secret and algorithm failed.'))
        req_data = {'client_id': self.client_id, 'client_assertion_type': 'urn:ietf:params:oauth:client-assertion-type:jwt-bearer', 'client_assertion': client_assertion, 'token': access_token, 'token_type_hint': 'access_token'}
        http_response = self.session.request(self.introspect_endpoint, 'POST', authenticated=False, data=req_data)
        return http_response