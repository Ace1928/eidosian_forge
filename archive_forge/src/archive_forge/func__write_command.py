from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def _write_command(self, cmd):
    """
        Write the given command to the Nagios command file
        """
    if not os.path.exists(self.cmdfile):
        self.module.fail_json(msg='nagios command file does not exist', cmdfile=self.cmdfile)
    if not stat.S_ISFIFO(os.stat(self.cmdfile).st_mode):
        self.module.fail_json(msg='nagios command file is not a fifo file', cmdfile=self.cmdfile)
    try:
        with open(self.cmdfile, 'w') as fp:
            fp.write(cmd)
            fp.flush()
        self.command_results.append(cmd.strip())
    except IOError:
        self.module.fail_json(msg='unable to write to nagios command file', cmdfile=self.cmdfile)