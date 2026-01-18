from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def unsilence_host(self, host):
    """
        This command is used to enable notifications for the host and
        all services on the specified host.

        This is equivalent to calling enable_host_svc_notifications
        and enable_host_notifications.

        Syntax: ENABLE_HOST_SVC_NOTIFICATIONS;<host_name>
        Syntax: ENABLE_HOST_NOTIFICATIONS;<host_name>
        """
    cmd = ['ENABLE_HOST_SVC_NOTIFICATIONS', 'ENABLE_HOST_NOTIFICATIONS']
    nagios_return = True
    return_str_list = []
    for c in cmd:
        notif_str = self._fmt_notif_str(c, host)
        nagios_return = self._write_command(notif_str) and nagios_return
        return_str_list.append(notif_str)
    if nagios_return:
        return return_str_list
    else:
        return 'Fail: could not write to the command file'