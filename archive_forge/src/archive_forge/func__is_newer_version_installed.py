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
def _is_newer_version_installed(self, pkg_name):
    candidate_pkg = self._packagename_dict(pkg_name)
    if not candidate_pkg:
        return False
    installed = self.base.sack.query().installed()
    installed_pkg = installed.filter(name=candidate_pkg['name']).run()
    if installed_pkg:
        installed_pkg = installed_pkg[0]
        evr_cmp = self._compare_evr(installed_pkg.epoch, installed_pkg.version, installed_pkg.release, candidate_pkg['epoch'], candidate_pkg['version'], candidate_pkg['release'])
        return evr_cmp == 1
    else:
        return False