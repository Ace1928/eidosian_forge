from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def _fmt_chk_str(self, cmd, host, svc=None, start=None):
    """
        Format an external-command forced host or service check string.

        cmd - Nagios command ID
        host - Host to check service from
        svc - Service to check
        start - check time

        Syntax: [submitted] COMMAND;<host_name>;[<service_description>];<check_time>
        """
    entry_time = self._now()
    hdr = '[%s] %s;%s;' % (entry_time, cmd, host)
    if start is None:
        start = entry_time + 3
    if svc is None:
        chk_args = [str(start)]
    else:
        chk_args = [svc, str(start)]
    chk_arg_str = ';'.join(chk_args)
    chk_str = hdr + chk_arg_str + '\n'
    return chk_str