from __future__ import (absolute_import, division, print_function)
import re
from abc import ABC, abstractmethod
from ansible.errors import AnsibleConnectionFailure
def _exec_cli_command(self, cmd, check_rc=True):
    """
        Executes the CLI command on the remote device and returns the output

        :arg cmd: Byte string command to be executed
        """
    return self._connection.exec_command(cmd)