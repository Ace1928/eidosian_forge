from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def addparam_network(self, c):
    if LooseVersion(self.podman_version) >= LooseVersion('4.0.0'):
        for net in self.params['network']:
            c += ['--network', net]
        return c
    return c + ['--network', ','.join(self.params['network'])]