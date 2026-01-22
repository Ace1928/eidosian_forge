from __future__ import absolute_import, division, print_function
import os
import errno
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.six.moves import shlex_quote
def _write_password_to_pipe(proc):
    os.close(_sshpass_pipe[0])
    try:
        os.write(_sshpass_pipe[1], to_bytes(rsync_password) + b'\n')
    except OSError as exc:
        if exc.errno != errno.EPIPE or proc.poll() is None:
            raise