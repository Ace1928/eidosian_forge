from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def unsilence_nagios(self):
    """
        This command is used to enable notifications for all hosts and services
        in nagios.

        This is a 'OK, NAGIOS, GO'' command
        """
    cmd = 'ENABLE_NOTIFICATIONS'
    self._write_command(self._fmt_notif_str(cmd))