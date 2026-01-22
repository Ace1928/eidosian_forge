from __future__ import absolute_import, division, print_function
import json
import os.path
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
def _upgrade_current_package(self):
    command = 'upgrade'
    if not self.valid_package(self.current_package):
        self.failed = True
        self.message = 'Invalid package: {0}.'.format(self.current_package)
        raise HomebrewException(self.message)
    if not self._current_package_is_installed():
        command = 'install'
    if self._current_package_is_installed() and (not self._current_package_is_outdated()):
        self.message = 'Package is already upgraded: {0}'.format(self.current_package)
        self.unchanged_count += 1
        self.unchanged_pkgs.append(self.current_package)
        return True
    if self.module.check_mode:
        self.changed = True
        self.message = 'Package would be upgraded: {0}'.format(self.current_package)
        raise HomebrewException(self.message)
    opts = [self.brew_path, command] + self.install_options + [self.current_package]
    cmd = [opt for opt in opts if opt]
    rc, out, err = self.module.run_command(cmd)
    if self._current_package_is_installed() and (not self._current_package_is_outdated()):
        self.changed_count += 1
        self.changed_pkgs.append(self.current_package)
        self.changed = True
        self.message = 'Package upgraded: {0}'.format(self.current_package)
        return True
    else:
        self.failed = True
        self.message = err.strip()
        raise HomebrewException(self.message)