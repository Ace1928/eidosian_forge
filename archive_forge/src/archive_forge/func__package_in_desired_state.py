from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
def _package_in_desired_state(self, name, want_installed, version=None):
    dummy, out, dummy = self.runner('state package').run(state='query', package=name)
    has_package = out.startswith(name + ' - %s' % ('' if not version else version + ' '))
    return want_installed == has_package