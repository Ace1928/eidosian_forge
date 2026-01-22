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
def _get_expected_values_platform(module, client, api_version, options, image, values, host_info):
    expected_values = {}
    if 'platform' in values:
        try:
            expected_values['platform'] = normalize_platform_string(values['platform'], daemon_os=host_info.get('OSType') if host_info else None, daemon_arch=host_info.get('Architecture') if host_info else None)
        except ValueError as exc:
            module.fail_json(msg='Error while parsing platform parameer: %s' % (to_native(exc),))
    return expected_values