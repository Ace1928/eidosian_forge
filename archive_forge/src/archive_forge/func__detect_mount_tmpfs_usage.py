from __future__ import absolute_import, division, print_function
import shlex
import time
import traceback
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.basic import human_to_bytes
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
def _detect_mount_tmpfs_usage(client):
    for mount in client.module.params['mounts'] or []:
        if mount.get('type') == 'tmpfs':
            return True
        if mount.get('tmpfs_size') is not None:
            return True
        if mount.get('tmpfs_mode') is not None:
            return True
    return False