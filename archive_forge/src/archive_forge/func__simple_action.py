from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def _simple_action(self):
    if self.action in ['start', 'restart', 'stop', 'pause', 'unpause', 'kill']:
        cmd = [self.action, self.params['name']]
        return [to_bytes(i, errors='surrogate_or_strict') for i in cmd]
    if self.action == 'delete':
        cmd = ['rm', '-f', self.params['name']]
        return [to_bytes(i, errors='surrogate_or_strict') for i in cmd]
    self.module.fail_json(msg='Unknown action %s' % self.action)