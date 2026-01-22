from __future__ import absolute_import, division, print_function
import json
import os.path
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
def _link_current_package(self):
    if not self.valid_package(self.current_package):
        self.failed = True
        self.message = 'Invalid package: {0}.'.format(self.current_package)
        raise HomebrewException(self.message)
    if not self._current_package_is_installed():
        self.failed = True
        self.message = 'Package not installed: {0}.'.format(self.current_package)
        raise HomebrewException(self.message)
    if self.module.check_mode:
        self.changed = True
        self.message = 'Package would be linked: {0}'.format(self.current_package)
        raise HomebrewException(self.message)
    opts = [self.brew_path, 'link'] + self.install_options + [self.current_package]
    cmd = [opt for opt in opts if opt]
    rc, out, err = self.module.run_command(cmd)
    if rc == 0:
        self.changed_count += 1
        self.changed_pkgs.append(self.current_package)
        self.changed = True
        self.message = 'Package linked: {0}'.format(self.current_package)
        return True
    else:
        self.failed = True
        self.message = 'Package could not be linked: {0}.'.format(self.current_package)
        raise HomebrewException(self.message)