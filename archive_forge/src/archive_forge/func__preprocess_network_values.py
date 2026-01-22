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
def _preprocess_network_values(module, client, api_version, options, values):
    if 'networks' in values:
        for network in values['networks']:
            network['id'] = _get_network_id(module, client, network['name'])
            if not network['id']:
                client.fail('Parameter error: network named %s could not be found. Does it exist?' % (network['name'],))
    if 'network_mode' in values:
        values['network_mode'] = _preprocess_container_names(module, client, api_version, values['network_mode'])
    return values