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
def _add_args(self, b_command: list[bytes], b_args: t.Iterable[bytes], explanation: str) -> None:
    """
        Adds arguments to the ssh command and displays a caller-supplied explanation of why.

        :arg b_command: A list containing the command to add the new arguments to.
            This list will be modified by this method.
        :arg b_args: An iterable of new arguments to add.  This iterable is used
            more than once so it must be persistent (ie: a list is okay but a
            StringIO would not)
        :arg explanation: A text string containing explaining why the arguments
            were added.  It will be displayed with a high enough verbosity.
        .. note:: This function does its work via side-effect.  The b_command list has the new arguments appended.
        """
    display.vvvvv(u'SSH: %s: (%s)' % (explanation, ')('.join((to_text(a) for a in b_args))), host=self.host)
    b_command += b_args