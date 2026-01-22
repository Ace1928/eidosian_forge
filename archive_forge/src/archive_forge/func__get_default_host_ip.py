from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils._platform import (
from ansible_collections.community.docker.plugins.module_utils.module_container.base import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._api.errors import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def _get_default_host_ip(module, client):
    if module.params['default_host_ip'] is not None:
        return module.params['default_host_ip']
    ip = '0.0.0.0'
    for network_data in module.params['networks'] or []:
        if network_data.get('name'):
            network = client.get_network(network_data['name'])
            if network is None:
                client.fail("Cannot inspect the network '{0}' to determine the default IP".format(network_data['name']))
            if network.get('Driver') == 'bridge' and network.get('Options', {}).get('com.docker.network.bridge.host_binding_ipv4'):
                ip = network['Options']['com.docker.network.bridge.host_binding_ipv4']
                break
    return ip