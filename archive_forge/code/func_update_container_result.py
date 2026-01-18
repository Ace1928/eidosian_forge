from __future__ import (absolute_import, division, print_function)
import json  # noqa: F402
import os  # noqa: F402
import shlex  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import normalize_signal
from ansible_collections.containers.podman.plugins.module_utils.podman.common import ARGUMENTS_OPTS_DICT
def update_container_result(self, changed=True):
    """Inspect the current container, update results with last info, exit.

        Keyword Arguments:
            changed {bool} -- whether any action was performed
                              (default: {True})
        """
    facts = self.container.get_info() if changed else self.container.info
    out, err = (self.container.stdout, self.container.stderr)
    self.results.update({'changed': changed, 'container': facts, 'podman_actions': self.container.actions}, stdout=out, stderr=err)
    if self.container.diff:
        self.results.update({'diff': self.container.diff})
    if self.module.params['debug'] or self.module_params['debug']:
        self.results.update({'podman_version': self.container.version})
    sysd = generate_systemd(self.module, self.module_params, self.name, self.container.version)
    self.results['changed'] = changed or sysd['changed']
    self.results.update({'podman_systemd': sysd['systemd']})
    if sysd['diff']:
        if 'diff' not in self.results:
            self.results.update({'diff': sysd['diff']})
        else:
            self.results['diff']['before'] += sysd['diff']['before']
            self.results['diff']['after'] += sysd['diff']['after']