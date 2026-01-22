from __future__ import absolute_import, division, print_function
import os
import re
import sys
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.urls import fetch_file
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
def _update_only(self, pkgs):
    not_installed = []
    for pkg in pkgs:
        if self._is_installed(self._package_dict(pkg)['nevra'] if isinstance(pkg, dnf.package.Package) else pkg):
            try:
                if isinstance(pkg, dnf.package.Package):
                    self.base.package_upgrade(pkg)
                else:
                    self.base.upgrade(pkg)
            except Exception as e:
                self.module.fail_json(msg='Error occurred attempting update_only operation: {0}'.format(to_native(e)), results=[], rc=1)
        else:
            not_installed.append(pkg)
    return not_installed