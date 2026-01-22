from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.facts.packages import LibMgr, CLIMgr, get_all_pkg_managers
class PACMAN(CLIMgr):
    CLI = 'pacman'

    def list_installed(self):
        locale = get_best_parsable_locale(module)
        rc, out, err = module.run_command([self._cli, '-Qi'], environ_update=dict(LC_ALL=locale))
        if rc != 0 or err:
            raise Exception('Unable to list packages rc=%s : %s' % (rc, err))
        return out.split('\n\n')[:-1]

    def get_package_details(self, package):
        raw_pkg_details = {}
        last_detail = None
        for line in package.splitlines():
            m = re.match('([\\w ]*[\\w]) +: (.*)', line)
            if m:
                last_detail = m.group(1)
                raw_pkg_details[last_detail] = m.group(2)
            else:
                raw_pkg_details[last_detail] = raw_pkg_details[last_detail] + '  ' + line.lstrip()
        provides = None
        if raw_pkg_details['Provides'] != 'None':
            provides = [p.split('=')[0] for p in raw_pkg_details['Provides'].split('  ')]
        return {'name': raw_pkg_details['Name'], 'version': raw_pkg_details['Version'], 'arch': raw_pkg_details['Architecture'], 'provides': provides}