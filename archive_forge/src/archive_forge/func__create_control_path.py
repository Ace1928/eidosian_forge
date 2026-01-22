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
def _create_control_path(host: str | None, port: int | None, user: str | None, connection: ConnectionBase | None=None, pid: int | None=None) -> str:
    """Make a hash for the controlpath based on con attributes"""
    pstring = '%s-%s-%s' % (host, port, user)
    if connection:
        pstring += '-%s' % connection
    if pid:
        pstring += '-%s' % to_text(pid)
    m = hashlib.sha1()
    m.update(to_bytes(pstring))
    digest = m.hexdigest()
    cpath = '%(directory)s/' + digest[:10]
    return cpath