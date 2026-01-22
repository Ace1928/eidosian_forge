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
def _get_image_binds(volumes):
    """
    Convert array of binds to array of strings with format host_path:container_path:mode

    :param volumes: array of bind dicts
    :return: array of strings
    """
    results = []
    if isinstance(volumes, dict):
        results += _get_bind_from_dict(volumes)
    elif isinstance(volumes, list):
        for vol in volumes:
            results += _get_bind_from_dict(vol)
    return results