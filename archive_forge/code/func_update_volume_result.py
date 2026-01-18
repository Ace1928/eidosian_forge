from __future__ import absolute_import, division, print_function
import json  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
def update_volume_result(self, changed=True):
    """Inspect the current volume, update results with last info, exit.

        Keyword Arguments:
            changed {bool} -- whether any action was performed
                              (default: {True})
        """
    facts = self.volume.get_info() if changed else self.volume.info
    out, err = (self.volume.stdout, self.volume.stderr)
    self.results.update({'changed': changed, 'volume': facts, 'podman_actions': self.volume.actions}, stdout=out, stderr=err)
    if self.volume.diff:
        self.results.update({'diff': self.volume.diff})
    if self.module.params['debug']:
        self.results.update({'podman_version': self.volume.version})
    self.module.exit_json(**self.results)