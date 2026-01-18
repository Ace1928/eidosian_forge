from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import os
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
import platform
def upgrade_all(self):
    """ Upgrades all installed apps and sets the correct result data """
    outdated = self.outdated()
    if not self.module.check_mode:
        self.check_signin()
        rc, out, err = self.run(['upgrade'])
        if rc != 0:
            self.module.fail_json(msg='Could not upgrade all apps: ' + out.rstrip())
    self.count_upgrade += len(outdated)