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
def _get_expected_values_mounts(module, client, api_version, options, image, values, host_info):
    expected_values = {}
    if 'mounts' in values:
        expected_values['mounts'] = values['mounts']
    expected_vols = dict()
    if image and image['Config'].get('Volumes'):
        expected_vols.update(image['Config'].get('Volumes'))
    if 'volumes' in values:
        for vol in values['volumes']:
            if ':' in vol:
                parts = vol.split(':')
                if len(parts) == 3:
                    continue
                if len(parts) == 2:
                    if not _is_volume_permissions(parts[1]):
                        continue
            expected_vols[vol] = {}
    if expected_vols:
        expected_values['volumes'] = expected_vols
    image_vols = []
    if image:
        image_vols = _get_image_binds(image['Config'].get('Volumes'))
    param_vols = []
    if 'volume_binds' in values:
        param_vols = values['volume_binds']
    expected_values['volume_binds'] = list(set(image_vols + param_vols))
    return expected_values