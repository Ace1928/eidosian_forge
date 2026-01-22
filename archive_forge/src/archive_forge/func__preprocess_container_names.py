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
def _preprocess_container_names(module, client, api_version, value):
    if value is None or not value.startswith('container:'):
        return value
    container_name = value[len('container:'):]
    container = client.get_container(container_name)
    if container is None:
        module.warn('Cannot find a container with name or ID "{0}"'.format(container_name))
        return value
    return 'container:{0}'.format(container['Id'])