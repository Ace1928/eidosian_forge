from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
def _prepare_base_cmd():
    _tout_cmd = module.get_bin_path('timeout', False)
    if _tout_cmd:
        cmd = ['timeout', '-s', '9', module.params['timeout'], _puppet_cmd(module)]
    else:
        cmd = ['puppet']
    return cmd