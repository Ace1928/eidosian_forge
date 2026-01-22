from __future__ import absolute_import, division, print_function
import json
import os.path
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
def _upgrade_all_packages(self):
    opts = [self.brew_path, 'upgrade'] + self.install_options
    cmd = [opt for opt in opts if opt]
    rc, out, err = self.module.run_command(cmd)
    if rc == 0:
        self.changed = True
        self.message = 'All packages upgraded.'
        return True
    else:
        self.failed = True
        self.message = err.strip()
        raise HomebrewException(self.message)