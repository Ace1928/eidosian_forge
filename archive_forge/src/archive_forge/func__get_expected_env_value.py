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
def _get_expected_env_value(module, client, api_version, image, value, sentry):
    expected_env = {}
    if image and image['Config'].get('Env'):
        for env_var in image['Config']['Env']:
            parts = env_var.split('=', 1)
            expected_env[parts[0]] = parts[1]
    if value and value is not sentry:
        for env_var in value:
            parts = env_var.split('=', 1)
            expected_env[parts[0]] = parts[1]
    param_env = []
    for key, env_value in expected_env.items():
        param_env.append('%s=%s' % (key, env_value))
    return param_env