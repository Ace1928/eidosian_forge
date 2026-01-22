from __future__ import (annotations, absolute_import, division, print_function)
import collections.abc as c
import errno
import fcntl
import hashlib
import io
import os
import pty
import re
import shlex
import subprocess
import time
import typing as t
from functools import wraps
from ansible.errors import (
from ansible.errors import AnsibleOptionsError
from ansible.module_utils.compat import selectors
from ansible.module_utils.six import PY3, text_type, binary_type
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.parsing.convert_bool import BOOLEANS, boolean
from ansible.plugins.connection import ConnectionBase, BUFSIZE
from ansible.plugins.shell.powershell import _parse_clixml
from ansible.utils.display import Display
from ansible.utils.path import unfrackpath, makedirs_safe
def _send_initial_data(self, fh: io.IOBase, in_data: bytes, ssh_process: subprocess.Popen) -> None:
    """
        Writes initial data to the stdin filehandle of the subprocess and closes
        it. (The handle must be closed; otherwise, for example, "sftp -b -" will
        just hang forever waiting for more commands.)
        """
    display.debug(u'Sending initial data')
    try:
        fh.write(to_bytes(in_data))
        fh.close()
    except (OSError, IOError) as e:
        time.sleep(0.001)
        ssh_process.poll()
        if getattr(ssh_process, 'returncode', None) is None:
            raise AnsibleConnectionFailure('Data could not be sent to remote host "%s". Make sure this host can be reached over ssh: %s' % (self.host, to_native(e)), orig_exc=e)
    display.debug(u'Sent initial data (%d bytes)' % len(in_data))