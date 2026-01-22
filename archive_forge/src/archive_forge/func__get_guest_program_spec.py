from __future__ import absolute_import, division, print_function
import re
from os.path import exists, getsize
from socket import gaierror
from ssl import SSLError
from time import sleep
import traceback
from ansible.errors import AnsibleError, AnsibleFileNotFound, AnsibleConnectionFailure
from ansible.module_utils._text import to_bytes, to_native
from ansible.plugins.connection import ConnectionBase
from ansible.module_utils.basic import missing_required_lib
def _get_guest_program_spec(self, cmd, stdout, stderr):
    guest_program_spec = vim.GuestProgramSpec()
    program_path, arguments = self._get_program_spec_program_path_and_arguments(cmd)
    arguments += ' 1> %s 2> %s' % (stdout, stderr)
    guest_program_spec.programPath = program_path
    guest_program_spec.arguments = arguments
    return guest_program_spec