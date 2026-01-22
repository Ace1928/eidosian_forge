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
class PrivateKeyJwtAuthClient(AbstractAuthClient):
    """Http client with the auth method 'private_key_jwt'."""

    def __init__(self, session, introspect_endpoint, audience, client_id, func_get_config_option, logger):
        super(PrivateKeyJwtAuthClient, self).__init__(session, introspect_endpoint, audience, client_id, func_get_config_option, logger)
        self.jwt_key_file = self.get_config_option('jwt_key_file', is_required=True)
        self.jwt_bearer_time_out = self.get_config_option('jwt_bearer_time_out', is_required=True)
        self.jwt_algorithm = self.get_config_option('jwt_algorithm', is_required=True)
        self.logger = logger

    def introspect(self, access_token):
        """Access the introspect API.

        Access the Introspect API to verify the access token by
        the auth method 'private_key_jwt'.
        """
        if not os.path.isfile(self.jwt_key_file):
            self.logger.critical('Configuration error. JWT key file is not a file. path: %s' % self.jwt_key_file)
            raise ConfigurationError(_('Configuration error. JWT key file is not a file.'))
        try:
            with open(self.jwt_key_file, 'r') as jwt_file:
                jwt_key = jwt_file.read()
        except Exception as e:
            self.logger.critical('Configuration error. Failed to read the JWT key file. %s', e)
            raise ConfigurationError(_('Configuration error. Failed to read the JWT key file.'))
        if not jwt_key:
            self.logger.critical('Configuration error. The JWT key file content is empty. path: %s' % self.jwt_key_file)
            raise ConfigurationError(_('Configuration error. The JWT key file content is empty.'))
        iat = round(time.time())
        try:
            client_assertion = jwt.encode(payload={'jti': str(uuid.uuid4()), 'iat': str(iat), 'exp': str(iat + self.jwt_bearer_time_out), 'iss': self.client_id, 'sub': self.client_id, 'aud': self.audience}, headers={'typ': 'JWT', 'alg': self.jwt_algorithm}, key=jwt_key, algorithm=self.jwt_algorithm)
        except Exception as e:
            self.logger.critical('Configuration error. JWT encoding with the specified JWT key file and algorithm failed. path: %s, algorithm: %s, error: %s' % (self.jwt_key_file, self.jwt_algorithm, e))
            raise ConfigurationError(_('Configuration error. JWT encoding with the specified JWT key file and algorithm failed.'))
        req_data = {'client_id': self.client_id, 'client_assertion_type': 'urn:ietf:params:oauth:client-assertion-type:jwt-bearer', 'client_assertion': client_assertion, 'token': access_token, 'token_type_hint': 'access_token'}
        http_response = self.session.request(self.introspect_endpoint, 'POST', authenticated=False, data=req_data)
        return http_response