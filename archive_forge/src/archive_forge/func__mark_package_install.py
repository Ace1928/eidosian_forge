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
def _mark_package_install(self, pkg_spec, upgrade=False):
    """Mark the package for install."""
    is_newer_version_installed = self._is_newer_version_installed(pkg_spec)
    is_installed = self._is_installed(pkg_spec)
    try:
        if is_newer_version_installed:
            if self.allow_downgrade:
                if upgrade:
                    if is_installed:
                        self.base.upgrade(pkg_spec)
                    else:
                        self.base.install(pkg_spec, strict=self.base.conf.strict)
                else:
                    self.base.install(pkg_spec, strict=self.base.conf.strict)
            else:
                pass
        elif is_installed:
            if upgrade:
                self.base.upgrade(pkg_spec)
            else:
                pass
        else:
            self.base.install(pkg_spec, strict=self.base.conf.strict)
        return {'failed': False, 'msg': '', 'failure': '', 'rc': 0}
    except dnf.exceptions.MarkingError as e:
        return {'failed': True, 'msg': 'No package {0} available.'.format(pkg_spec), 'failure': ' '.join((pkg_spec, to_native(e))), 'rc': 1, 'results': []}
    except dnf.exceptions.DepsolveError as e:
        return {'failed': True, 'msg': 'Depsolve Error occurred for package {0}.'.format(pkg_spec), 'failure': ' '.join((pkg_spec, to_native(e))), 'rc': 1, 'results': []}
    except dnf.exceptions.Error as e:
        if to_text('already installed') in to_text(e):
            return {'failed': False, 'msg': '', 'failure': ''}
        else:
            return {'failed': True, 'msg': 'Unknown Error occurred for package {0}.'.format(pkg_spec), 'failure': ' '.join((pkg_spec, to_native(e))), 'rc': 1, 'results': []}