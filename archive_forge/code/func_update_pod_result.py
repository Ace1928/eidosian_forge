from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def update_pod_result(self, changed=True):
    """Inspect the current pod, update results with last info, exit.

        Keyword Arguments:
            changed {bool} -- whether any action was performed
                              (default: {True})
        """
    facts = self.pod.get_info() if changed else self.pod.info
    out, err = (self.pod.stdout, self.pod.stderr)
    self.results.update({'changed': changed, 'pod': facts, 'podman_actions': self.pod.actions}, stdout=out, stderr=err)
    if self.pod.diff:
        self.results.update({'diff': self.pod.diff})
    if self.module.params['debug'] or self.module_params['debug']:
        self.results.update({'podman_version': self.pod.version})
    sysd = generate_systemd(self.module, self.module_params, self.name, self.pod.version)
    self.results['changed'] = changed or sysd['changed']
    self.results.update({'podman_systemd': sysd['systemd']})
    if sysd['diff']:
        if 'diff' not in self.results:
            self.results.update({'diff': sysd['diff']})
        else:
            self.results['diff']['before'] += sysd['diff']['before']
            self.results['diff']['after'] += sysd['diff']['after']