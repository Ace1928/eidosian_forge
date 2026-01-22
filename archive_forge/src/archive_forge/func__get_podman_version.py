from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def _get_podman_version(self):
    rc, out, err = self.module.run_command([self.module_params['executable'], b'--version'])
    if rc != 0 or not out or 'version' not in out:
        self.module.fail_json(msg='%s run failed!' % self.module_params['executable'])
    return out.split('version')[1].strip()