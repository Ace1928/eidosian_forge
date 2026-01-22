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
def _get_values_mounts(module, container, api_version, options, image, host_info):
    volumes = container['Config'].get('Volumes')
    binds = container['HostConfig'].get('Binds')
    mounts = container['HostConfig'].get('Mounts')
    if mounts is not None:
        result = []
        empty_dict = {}
        for mount in mounts:
            result.append({'type': mount.get('Type'), 'source': mount.get('Source'), 'target': mount.get('Target'), 'read_only': mount.get('ReadOnly', False), 'consistency': mount.get('Consistency'), 'propagation': mount.get('BindOptions', empty_dict).get('Propagation'), 'no_copy': mount.get('VolumeOptions', empty_dict).get('NoCopy', False), 'labels': mount.get('VolumeOptions', empty_dict).get('Labels', empty_dict), 'volume_driver': mount.get('VolumeOptions', empty_dict).get('DriverConfig', empty_dict).get('Name'), 'volume_options': mount.get('VolumeOptions', empty_dict).get('DriverConfig', empty_dict).get('Options', empty_dict), 'tmpfs_size': mount.get('TmpfsOptions', empty_dict).get('SizeBytes'), 'tmpfs_mode': mount.get('TmpfsOptions', empty_dict).get('Mode')})
        mounts = result
    result = {}
    if volumes is not None:
        result['volumes'] = volumes
    if binds is not None:
        result['volume_binds'] = binds
    if mounts is not None:
        result['mounts'] = mounts
    return result