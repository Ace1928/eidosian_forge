from __future__ import (annotations, absolute_import, division, print_function)
import base64
import json
import logging
import os
import typing as t
from ansible import constants as C
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible.errors import AnsibleFileNotFound
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.plugins.connection import ConnectionBase
from ansible.plugins.shell.powershell import ShellModule as PowerShellPlugin
from ansible.plugins.shell.powershell import _common_args
from ansible.utils.display import Display
from ansible.utils.hashing import sha1
def _exec_psrp_script(self, script: str, input_data: bytes | str | t.Iterable | None=None, use_local_scope: bool=True, arguments: t.Iterable[str] | None=None) -> tuple[int, bytes, bytes]:
    if self._last_pipeline:
        self._last_pipeline.state = PSInvocationState.RUNNING
        self._last_pipeline.stop()
        self._last_pipeline = None
    ps = PowerShell(self.runspace)
    ps.add_script(script, use_local_scope=use_local_scope)
    if arguments:
        for arg in arguments:
            ps.add_argument(arg)
    ps.invoke(input=input_data)
    rc, stdout, stderr = self._parse_pipeline_result(ps)
    self._last_pipeline = ps
    return (rc, stdout, stderr)