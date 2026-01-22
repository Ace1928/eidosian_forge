import os
import requests
from keystoneauth1 import loading
from keystoneauth1 import plugin
from oslo_log import log
class VitrageKeycloakPlugin(plugin.BaseAuthPlugin):
    """Authentication plugin for Keycloak """

    def __init__(self, username, password, realm_name, endpoint, auth_url, openid_client_id):
        self.username = username
        self.password = password
        self.realm_name = realm_name
        self.endpoint = endpoint
        self.auth_url = auth_url
        self.client_id = openid_client_id
        self.verify = True

    def get_headers(self, session, **kwargs):
        self.verify = session.verify
        return {'X-Auth-Token': self._authenticate_keycloak(), 'x-user-id': self.username, 'x-project-id': self.realm_name}

    def get_endpoint(self, session, **kwargs):
        return self.endpoint

    def _authenticate_keycloak(self):
        keycloak_endpoint = '%s/realms/%s/protocol/openid-connect/token' % (self.auth_url, self.realm_name)
        body = {'grant_type': 'password', 'username': self.username, 'password': self.password, 'client_id': self.client_id, 'scope': 'profile'}
        resp = requests.post(keycloak_endpoint, data=body, verify=self.verify)
        try:
            resp.raise_for_status()
        except Exception as e:
            LOG.error('Failed to get access token: %s', str(e))
        return resp.json()['access_token']