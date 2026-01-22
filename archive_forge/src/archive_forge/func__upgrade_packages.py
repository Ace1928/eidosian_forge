from __future__ import absolute_import, division, print_function
import json
import os.path
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
def _upgrade_packages(self):
    if not self.packages:
        self._upgrade_all_packages()
    else:
        for package in self.packages:
            self.current_package = package
            self._upgrade_current_package()
        return True