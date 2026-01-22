from __future__ import absolute_import, division, print_function
import re
import json
import numbers
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.snap import snap_runner
def _present(self, actionable_snaps, refresh=False):
    self.changed = True
    self.vars.snaps_installed = actionable_snaps
    if self.check_mode:
        return
    params = ['state', 'classic', 'channel', 'dangerous']
    has_one_pkg_params = bool(self.vars.classic) or self.vars.channel != 'stable'
    has_multiple_snaps = len(actionable_snaps) > 1
    if has_one_pkg_params and has_multiple_snaps:
        self.vars.cmd, rc, out, err, run_info = self._run_multiple_commands(params, actionable_snaps, bundle=False, refresh=refresh)
    else:
        self.vars.cmd, rc, out, err, run_info = self._run_multiple_commands(params, actionable_snaps, refresh=refresh)
    self.vars.run_info = run_info
    if rc == 0:
        return
    classic_snap_pattern = re.compile('^error: This revision of snap "(?P<package_name>\\w+)" was published using classic confinement')
    match = classic_snap_pattern.match(err)
    if match:
        err_pkg = match.group('package_name')
        msg = "Couldn't install {name} because it requires classic confinement".format(name=err_pkg)
    else:
        msg = "Ooops! Snap installation failed while executing '{cmd}', please examine logs and error output for more details.".format(cmd=self.vars.cmd)
    self.do_raise(msg=msg)