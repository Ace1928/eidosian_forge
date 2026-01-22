from __future__ import absolute_import, division, print_function
import re
import shlex
from ansible.module_utils.basic import AnsibleModule
from collections import defaultdict, namedtuple
def _install_packages_for_real(pacman_verb, pkglist):
    cmd = cmd_base + [pacman_verb] + [p.source for p in pkglist]
    rc, stdout, stderr = self.m.run_command(cmd, check_rc=False)
    if rc != 0:
        self.fail('Failed to install package(s)', cmd=cmd, stdout=stdout, stderr=stderr)
    self.add_exit_infos(stdout=stdout, stderr=stderr)
    self._invalidate_database()