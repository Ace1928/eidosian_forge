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
@_ssh_retry
def _file_transport_command(self, in_path: str, out_path: str, sftp_action: str) -> tuple[int, bytes, bytes]:
    host = '[%s]' % self.host
    smart_methods = ['sftp', 'scp', 'piped']
    if getattr(self._shell, '_IS_WINDOWS', False):
        smart_methods.remove('piped')
    methods = []
    ssh_transfer_method = self.get_option('ssh_transfer_method')
    scp_if_ssh = self.get_option('scp_if_ssh')
    if ssh_transfer_method is None and scp_if_ssh == 'smart':
        ssh_transfer_method = 'smart'
    if ssh_transfer_method is not None:
        if ssh_transfer_method == 'smart':
            methods = smart_methods
        else:
            methods = [ssh_transfer_method]
    else:
        if not isinstance(scp_if_ssh, bool):
            scp_if_ssh = scp_if_ssh.lower()
            if scp_if_ssh in BOOLEANS:
                scp_if_ssh = boolean(scp_if_ssh, strict=False)
            elif scp_if_ssh != 'smart':
                raise AnsibleOptionsError('scp_if_ssh needs to be one of [smart|True|False]')
        if scp_if_ssh == 'smart':
            methods = smart_methods
        elif scp_if_ssh is True:
            methods = ['scp']
        else:
            methods = ['sftp']
    for method in methods:
        returncode = stdout = stderr = None
        if method == 'sftp':
            cmd = self._build_command(self.get_option('sftp_executable'), 'sftp', to_bytes(host))
            in_data = u'{0} {1} {2}\n'.format(sftp_action, shlex.quote(in_path), shlex.quote(out_path))
            in_data = to_bytes(in_data, nonstring='passthru')
            returncode, stdout, stderr = self._bare_run(cmd, in_data, checkrc=False)
        elif method == 'scp':
            scp = self.get_option('scp_executable')
            if sftp_action == 'get':
                cmd = self._build_command(scp, 'scp', u'{0}:{1}'.format(host, self._shell.quote(in_path)), out_path)
            else:
                cmd = self._build_command(scp, 'scp', in_path, u'{0}:{1}'.format(host, self._shell.quote(out_path)))
            in_data = None
            returncode, stdout, stderr = self._bare_run(cmd, in_data, checkrc=False)
        elif method == 'piped':
            if sftp_action == 'get':
                returncode, stdout, stderr = self.exec_command('dd if=%s bs=%s' % (in_path, BUFSIZE), sudoable=False)
                with open(to_bytes(out_path, errors='surrogate_or_strict'), 'wb+') as out_file:
                    out_file.write(stdout)
            else:
                with open(to_bytes(in_path, errors='surrogate_or_strict'), 'rb') as f:
                    in_data = to_bytes(f.read(), nonstring='passthru')
                if not in_data:
                    count = ' count=0'
                else:
                    count = ''
                returncode, stdout, stderr = self.exec_command('dd of=%s bs=%s%s' % (out_path, BUFSIZE, count), in_data=in_data, sudoable=False)
        if returncode == 0:
            return (returncode, stdout, stderr)
        elif len(methods) > 1:
            display.warning(u'%s transfer mechanism failed on %s. Use ANSIBLE_DEBUG=1 to see detailed information' % (method, host))
            display.debug(u'%s' % to_text(stdout))
            display.debug(u'%s' % to_text(stderr))
    if returncode == 255:
        raise AnsibleConnectionFailure('Failed to connect to the host via %s: %s' % (method, to_native(stderr)))
    else:
        raise AnsibleError('failed to transfer file to %s %s:\n%s\n%s' % (to_native(in_path), to_native(out_path), to_native(stdout), to_native(stderr)))