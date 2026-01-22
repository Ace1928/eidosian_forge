from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def acknowledge_host_problem(self, host):
    """
        This command is used to acknowledge a particular
        host problem.

        By acknowledging the current problem, future notifications
        for the same servicestate are disabled

        Syntax: ACKNOWLEDGE_HOST_PROBLEM;<host_name>;<sticky>;<notify>;
        <persistent>;<author>;<comment>
        """
    cmd = 'ACKNOWLEDGE_HOST_PROBLEM'
    ack_cmd_str = self._fmt_ack_str(cmd, host)
    self._write_command(ack_cmd_str)