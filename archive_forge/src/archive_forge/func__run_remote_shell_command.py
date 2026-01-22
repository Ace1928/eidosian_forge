import os
import re
import time
import logging
import warnings
import subprocess
from typing import List, Type, Tuple, Union, Optional, cast
from os.path import join as pjoin
from os.path import split as psplit
from libcloud.utils.py3 import StringIO, b
from libcloud.utils.logging import ExtraLogFormatter
def _run_remote_shell_command(self, cmd):
    """
        Run a command on a remote server.

        :param      cmd: Command to run.
        :type       cmd: ``list`` of ``str``

        :return: Command stdout, stderr and status code.
        :rtype: ``tuple``
        """
    base_cmd = self._get_base_ssh_command()
    full_cmd = base_cmd + [' '.join(cmd)]
    self.logger.debug('Executing command: "%s"' % ' '.join(full_cmd))
    child = subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = child.communicate()
    stdout_str = cast(str, stdout)
    stderr_str = cast(str, stdout)
    return (stdout_str, stderr_str, child.returncode)