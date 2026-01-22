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
def _preprocess_devices(module, client, api_version, value):
    if not value:
        return value
    expected_devices = []
    for device in value:
        parts = device.split(':')
        if len(parts) == 1:
            expected_devices.append(dict(CgroupPermissions='rwm', PathInContainer=parts[0], PathOnHost=parts[0]))
        elif len(parts) == 2:
            parts = device.split(':')
            expected_devices.append(dict(CgroupPermissions='rwm', PathInContainer=parts[1], PathOnHost=parts[0]))
        else:
            expected_devices.append(dict(CgroupPermissions=parts[2], PathInContainer=parts[1], PathOnHost=parts[0]))
    return expected_devices