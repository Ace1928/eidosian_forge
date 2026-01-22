import os
from troveclient import base
from troveclient import common
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules as core_modules
from swiftclient import client as swift_client
def _get_swift_client(self):
    if hasattr(self.api.client, 'auth'):
        auth_url = self.api.client.auth.auth_url
        user = self.api.client.auth._username
        key = self.api.client.auth._password
        tenant_name = self.api.client.auth._project_name
        auth_version = '3.0'
    else:
        auth_url = self.api.client.auth_url
        user = self.api.client.username
        key = self.api.client.password
        tenant_name = self.api.client.tenant
        auth_version = '2.0'
    token_str = '/tokens'
    if auth_url.endswith(token_str):
        auth_url = auth_url[:-len(token_str)]
    region_name = self.api.client.region_name
    os_options = {'tenant_name': tenant_name, 'region_name': region_name}
    return swift_client.Connection(auth_url, user, key, auth_version=auth_version, os_options=os_options)