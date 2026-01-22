from __future__ import (annotations, absolute_import, division, print_function)
import base64
import logging
import os
import re
import traceback
import json
import tempfile
import shlex
import subprocess
import time
import typing as t
import xml.etree.ElementTree as ET
from inspect import getfullargspec
from urllib.parse import urlunsplit
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.errors import AnsibleFileNotFound
from ansible.module_utils.json_utils import _filter_non_json_lines
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.plugins.connection import ConnectionBase
from ansible.plugins.shell.powershell import _parse_clixml
from ansible.plugins.shell.powershell import ShellBase as PowerShellBase
from ansible.utils.hashing import secure_hash
from ansible.utils.display import Display
def _winrm_exec(self, command: str, args: t.Iterable[bytes]=(), from_exec: bool=False, stdin_iterator: t.Iterable[tuple[bytes, bool]]=None) -> tuple[int, bytes, bytes]:
    if not self.protocol:
        self.protocol = self._winrm_connect()
        self._connected = True
    if from_exec:
        display.vvvvv('WINRM EXEC %r %r' % (command, args), host=self._winrm_host)
    else:
        display.vvvvvv('WINRM EXEC %r %r' % (command, args), host=self._winrm_host)
    command_id = None
    try:
        stdin_push_failed = False
        command_id = self.protocol.run_command(self.shell_id, to_bytes(command), map(to_bytes, args), console_mode_stdin=stdin_iterator is None)
        try:
            if stdin_iterator:
                self._winrm_write_stdin(command_id, stdin_iterator)
        except Exception as ex:
            display.warning('ERROR DURING WINRM SEND INPUT - attempting to recover: %s %s' % (type(ex).__name__, to_text(ex)))
            display.debug(traceback.format_exc())
            stdin_push_failed = True
        b_stdout, b_stderr, rc = self._winrm_get_command_output(self.protocol, self.shell_id, command_id, try_once=stdin_push_failed)
        stdout = to_text(b_stdout)
        stderr = to_text(b_stderr)
        if from_exec:
            display.vvvvv('WINRM RESULT <Response code %d, out %r, err %r>' % (rc, stdout, stderr), host=self._winrm_host)
        display.vvvvvv('WINRM RC %d' % rc, host=self._winrm_host)
        display.vvvvvv('WINRM STDOUT %s' % stdout, host=self._winrm_host)
        display.vvvvvv('WINRM STDERR %s' % stderr, host=self._winrm_host)
        if b_stderr.startswith(b'#< CLIXML'):
            b_stderr = _parse_clixml(b_stderr)
            stderr = to_text(stderr)
        if stdin_push_failed:
            try:
                filtered_output, dummy = _filter_non_json_lines(stdout)
                json.loads(filtered_output)
            except ValueError:
                raise AnsibleError(f'winrm send_input failed; \nstdout: {stdout}\nstderr {stderr}')
        return (rc, b_stdout, b_stderr)
    except requests.exceptions.Timeout as exc:
        raise AnsibleConnectionFailure('winrm connection error: %s' % to_native(exc))
    finally:
        if command_id:
            try:
                self.protocol.cleanup_command(self.shell_id, command_id)
            except WinRMTransportError as e:
                if e.code != 400:
                    raise
                display.warning('Failed to cleanup running WinRM command, resources might still be in use on the target server')