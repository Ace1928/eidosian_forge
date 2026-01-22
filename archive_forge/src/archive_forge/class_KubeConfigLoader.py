import atexit
import base64
import copy
import datetime
import json
import logging
import os
import platform
import tempfile
import time
import google.auth
import google.auth.transport.requests
import oauthlib.oauth2
import urllib3
from ruamel import yaml
from requests_oauthlib import OAuth2Session
from six import PY3
from kubernetes.client import ApiClient, Configuration
from kubernetes.config.exec_provider import ExecProvider
from .config_exception import ConfigException
from .dateutil import UTC, format_rfc3339, parse_rfc3339
class KubeConfigLoader(object):

    def __init__(self, config_dict, active_context=None, get_google_credentials=None, config_base_path='', config_persister=None):
        if isinstance(config_dict, ConfigNode):
            self._config = config_dict
        else:
            self._config = ConfigNode('kube-config', config_dict)
        self._current_context = None
        self._user = None
        self._cluster = None
        self.set_active_context(active_context)
        self._config_base_path = config_base_path
        self._config_persister = config_persister

        def _refresh_credentials():
            credentials, project_id = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/userinfo.email'])
            request = google.auth.transport.requests.Request()
            credentials.refresh(request)
            return credentials
        if get_google_credentials:
            self._get_google_credentials = get_google_credentials
        else:
            self._get_google_credentials = _refresh_credentials

    def set_active_context(self, context_name=None):
        if context_name is None:
            context_name = self._config['current-context']
        self._current_context = self._config['contexts'].get_with_name(context_name)
        if self._current_context['context'].safe_get('user') and self._config.safe_get('users'):
            user = self._config['users'].get_with_name(self._current_context['context']['user'], safe=True)
            if user:
                self._user = user['user']
            else:
                self._user = None
        else:
            self._user = None
        self._cluster = self._config['clusters'].get_with_name(self._current_context['context']['cluster'])['cluster']

    def _load_authentication(self):
        """Read authentication from kube-config user section if exists.

        This function goes through various authentication methods in user
        section of kube-config and stops if it finds a valid authentication
        method. The order of authentication methods is:

            1. auth-provider (gcp, azure, oidc)
            2. token field (point to a token file)
            3. exec provided plugin
            4. username/password
        """
        if not self._user:
            return
        if self._load_auth_provider_token():
            return
        if self._load_user_token():
            return
        if self._load_from_exec_plugin():
            return
        self._load_user_pass_token()

    def _load_auth_provider_token(self):
        if 'auth-provider' not in self._user:
            return
        provider = self._user['auth-provider']
        if 'name' not in provider:
            return
        if provider['name'] == 'gcp':
            return self._load_gcp_token(provider)
        if provider['name'] == 'azure':
            return self._load_azure_token(provider)
        if provider['name'] == 'oidc':
            return self._load_oid_token(provider)

    def _load_azure_token(self, provider):
        if 'config' not in provider:
            return
        if 'access-token' not in provider['config']:
            return
        if 'expires-on' in provider['config']:
            if int(provider['config']['expires-on']) < time.gmtime():
                self._refresh_azure_token(provider['config'])
        self.token = 'Bearer %s' % provider['config']['access-token']
        return self.token

    def _refresh_azure_token(self, config):
        if 'adal' not in globals():
            raise ImportError('refresh token error, adal library not imported')
        tenant = config['tenant-id']
        authority = 'https://login.microsoftonline.com/{}'.format(tenant)
        context = adal.AuthenticationContext(authority, validate_authority=True)
        refresh_token = config['refresh-token']
        client_id = config['client-id']
        token_response = context.acquire_token_with_refresh_token(refresh_token, client_id, '00000002-0000-0000-c000-000000000000')
        provider = self._user['auth-provider']['config']
        provider.value['access-token'] = token_response['accessToken']
        provider.value['expires-on'] = token_response['expiresOn']
        if self._config_persister:
            self._config_persister(self._config.value)

    def _load_gcp_token(self, provider):
        if 'config' not in provider or 'access-token' not in provider['config'] or ('expiry' in provider['config'] and _is_expired(provider['config']['expiry'])):
            self._refresh_gcp_token()
        self.token = 'Bearer %s' % provider['config']['access-token']
        return self.token

    def _refresh_gcp_token(self):
        if 'config' not in self._user['auth-provider']:
            self._user['auth-provider'].value['config'] = {}
        provider = self._user['auth-provider']['config']
        credentials = self._get_google_credentials()
        provider.value['access-token'] = credentials.token
        provider.value['expiry'] = format_rfc3339(credentials.expiry)
        if self._config_persister:
            self._config_persister(self._config.value)

    def _load_oid_token(self, provider):
        if 'config' not in provider:
            return
        reserved_characters = frozenset(['=', '+', '/'])
        token = provider['config']['id-token']
        if any((char in token for char in reserved_characters)):
            return
        parts = token.split('.')
        if len(parts) != 3:
            return
        padding = (4 - len(parts[1]) % 4) * '='
        if len(padding) == 3:
            return
        if PY3:
            jwt_attributes = json.loads(base64.b64decode(parts[1] + padding).decode('utf-8'))
        else:
            jwt_attributes = json.loads(base64.b64decode(parts[1] + padding))
        expire = jwt_attributes.get('exp')
        if expire is not None and _is_expired(datetime.datetime.fromtimestamp(expire, tz=UTC)):
            self._refresh_oidc(provider)
            if self._config_persister:
                self._config_persister(self._config.value)
        self.token = 'Bearer %s' % provider['config']['id-token']
        return self.token

    def _refresh_oidc(self, provider):
        config = Configuration()
        if 'idp-certificate-authority-data' in provider['config']:
            ca_cert = tempfile.NamedTemporaryFile(delete=True)
            if PY3:
                cert = base64.b64decode(provider['config']['idp-certificate-authority-data']).decode('utf-8')
            else:
                cert = base64.b64decode(provider['config']['idp-certificate-authority-data'] + '==')
            with open(ca_cert.name, 'w') as fh:
                fh.write(cert)
            config.ssl_ca_cert = ca_cert.name
        else:
            config.verify_ssl = False
        client = ApiClient(configuration=config)
        response = client.request(method='GET', url='%s/.well-known/openid-configuration' % provider['config']['idp-issuer-url'])
        if response.status != 200:
            return
        response = json.loads(response.data)
        request = OAuth2Session(client_id=provider['config']['client-id'], token=provider['config']['refresh-token'], auto_refresh_kwargs={'client_id': provider['config']['client-id'], 'client_secret': provider['config']['client-secret']}, auto_refresh_url=response['token_endpoint'])
        try:
            refresh = request.refresh_token(token_url=response['token_endpoint'], refresh_token=provider['config']['refresh-token'], auth=(provider['config']['client-id'], provider['config']['client-secret']), verify=config.ssl_ca_cert if config.verify_ssl else None)
        except oauthlib.oauth2.rfc6749.errors.InvalidClientIdError:
            return
        provider['config'].value['id-token'] = refresh['id_token']
        provider['config'].value['refresh-token'] = refresh['refresh_token']

    def _load_from_exec_plugin(self):
        if 'exec' not in self._user:
            return
        try:
            status = ExecProvider(self._user['exec']).run()
            if 'token' not in status:
                logging.error('exec: missing token field in plugin output')
                return None
            self.token = 'Bearer %s' % status['token']
            return True
        except Exception as e:
            logging.error(str(e))

    def _load_user_token(self):
        base_path = self._get_base_path(self._user.path)
        token = FileOrData(self._user, 'tokenFile', 'token', file_base_path=base_path, base64_file_content=False).as_data()
        if token:
            self.token = 'Bearer %s' % token
            return True

    def _load_user_pass_token(self):
        if 'username' in self._user and 'password' in self._user:
            self.token = urllib3.util.make_headers(basic_auth=self._user['username'] + ':' + self._user['password']).get('authorization')
            return True

    def _get_base_path(self, config_path):
        if self._config_base_path is not None:
            return self._config_base_path
        if config_path is not None:
            return os.path.abspath(os.path.dirname(config_path))
        return ''

    def _load_cluster_info(self):
        if 'server' in self._cluster:
            self.host = self._cluster['server'].rstrip('/')
            if self.host.startswith('https'):
                base_path = self._get_base_path(self._cluster.path)
                self.ssl_ca_cert = FileOrData(self._cluster, 'certificate-authority', file_base_path=base_path).as_file()
                self.cert_file = FileOrData(self._user, 'client-certificate', file_base_path=base_path).as_file()
                self.key_file = FileOrData(self._user, 'client-key', file_base_path=base_path).as_file()
        if 'insecure-skip-tls-verify' in self._cluster:
            self.verify_ssl = not self._cluster['insecure-skip-tls-verify']

    def _using_gcp_auth_provider(self):
        return self._user and 'auth-provider' in self._user and ('name' in self._user['auth-provider']) and (self._user['auth-provider']['name'] == 'gcp')

    def _set_config(self, client_configuration):
        if self._using_gcp_auth_provider():

            def _gcp_get_api_key(*args):
                return self._load_gcp_token(self._user['auth-provider'])
            client_configuration.get_api_key_with_prefix = _gcp_get_api_key
        if 'token' in self.__dict__:
            client_configuration.api_key['authorization'] = self.token
        keys = ['host', 'ssl_ca_cert', 'cert_file', 'key_file', 'verify_ssl']
        for key in keys:
            if key in self.__dict__:
                setattr(client_configuration, key, getattr(self, key))

    def load_and_set(self, client_configuration):
        self._load_authentication()
        self._load_cluster_info()
        self._set_config(client_configuration)

    def list_contexts(self):
        return [context.value for context in self._config['contexts']]

    @property
    def current_context(self):
        return self._current_context.value