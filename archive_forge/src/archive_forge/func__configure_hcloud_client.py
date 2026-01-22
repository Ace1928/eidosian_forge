from __future__ import annotations
import os
import sys
from ipaddress import IPv6Network
from ansible.errors import AnsibleError
from ansible.inventory.manager import InventoryData
from ansible.module_utils.common.text.converters import to_native
from ansible.plugins.inventory import BaseInventoryPlugin, Cacheable, Constructable
from ansible.utils.display import Display
from ..module_utils.client import (
from ..module_utils.vendor.hcloud import APIException
from ..module_utils.vendor.hcloud.networks import Network
from ..module_utils.vendor.hcloud.servers import Server
from ..module_utils.version import version
def _configure_hcloud_client(self):
    api_token_env = self.get_option('api_token_env')
    if api_token_env != 'HCLOUD_TOKEN':
        self.display.deprecated("The 'api_token_env' option is deprecated, please use the 'HCLOUD_TOKEN' environment variable or use the 'ansible.builtin.env' lookup instead.", version='3.0.0', collection_name='hetzner.hcloud')
        if api_token_env in os.environ:
            self.set_option('api_token', os.environ.get(api_token_env))
    api_token = self.get_option('api_token')
    api_endpoint = self.get_option('api_endpoint')
    if api_token is None:
        raise AnsibleError('No setting was provided for required configuration setting: plugin_type: inventory plugin: hetzner.hcloud.hcloud setting: api_token')
    api_token = self.templar.template(api_token)
    self.client = Client(token=api_token, api_endpoint=api_endpoint, application_name='ansible-inventory', application_version=version)
    try:
        self.client.locations.get_list()
    except APIException as exception:
        raise AnsibleError('Invalid Hetzner Cloud API Token.') from exception