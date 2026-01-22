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
def _preprocess_device_requests(module, client, api_version, value):
    if not value:
        return value
    device_requests = []
    for dr in value:
        device_requests.append({'Driver': dr['driver'], 'Count': dr['count'], 'DeviceIDs': dr['device_ids'], 'Capabilities': dr['capabilities'], 'Options': dr['options']})
    return device_requests