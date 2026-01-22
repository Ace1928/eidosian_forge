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
class ClientSecretBasicAuthClient(AbstractAuthClient):
    """Http client with the auth method 'client_secret_basic'."""

    def __init__(self, session, introspect_endpoint, audience, client_id, func_get_config_option, logger):
        super(ClientSecretBasicAuthClient, self).__init__(session, introspect_endpoint, audience, client_id, func_get_config_option, logger)
        self.client_secret = self.get_config_option('client_secret', is_required=True)

    def introspect(self, access_token):
        """Access the introspect API.

        Access the Introspect API to verify the access token by
        the auth method 'client_secret_basic'.
        """
        req_data = {'token': access_token, 'token_type_hint': 'access_token'}
        auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
        http_response = self.session.request(self.introspect_endpoint, 'POST', authenticated=False, data=req_data, requests_auth=auth)
        return http_response