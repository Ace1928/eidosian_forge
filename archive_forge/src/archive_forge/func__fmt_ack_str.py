from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def _fmt_ack_str(self, cmd, host, author=None, comment=None, svc=None, sticky=0, notify=1, persistent=0):
    """
        Format an external-command acknowledge string.

        cmd - Nagios command ID
        host - Host schedule downtime on
        author - Name to file the downtime as
        comment - Reason for running this command (upgrade, reboot, etc)
        svc - Service to schedule downtime for, omit when for host downtime
        sticky - the acknowledgement will remain until the host returns to an UP state if set to 1
        notify -  a notification will be sent out to contacts
        persistent - survive across restarts of the Nagios process

        Syntax: [submitted] COMMAND;<host_name>;[<service_description>]
        <sticky>;<notify>;<persistent>;<author>;<comment>
        """
    entry_time = self._now()
    hdr = '[%s] %s;%s;' % (entry_time, cmd, host)
    if not author:
        author = self.author
    if not comment:
        comment = self.comment
    if svc is not None:
        ack_args = [svc, str(sticky), str(notify), str(persistent), author, comment]
    else:
        ack_args = [str(sticky), str(notify), str(persistent), author, comment]
    ack_arg_str = ';'.join(ack_args)
    ack_str = hdr + ack_arg_str + '\n'
    return ack_str