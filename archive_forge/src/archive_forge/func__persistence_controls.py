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
@staticmethod
def _persistence_controls(b_command: list[bytes]) -> tuple[bool, bool]:
    """
        Takes a command array and scans it for ControlPersist and ControlPath
        settings and returns two booleans indicating whether either was found.
        This could be smarter, e.g. returning false if ControlPersist is 'no',
        but for now we do it simple way.
        """
    controlpersist = False
    controlpath = False
    for b_arg in (a.lower() for a in b_command):
        if b'controlpersist' in b_arg:
            controlpersist = True
        elif b'controlpath' in b_arg:
            controlpath = True
    return (controlpersist, controlpath)