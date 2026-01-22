from __future__ import absolute_import, division, print_function
import sys
import __main__
import atexit
import errno
import datetime
import grp
import fcntl
import locale
import os
import pwd
import platform
import re
import select
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import traceback
import types
from itertools import chain, repeat
from ansible.module_utils.compat import selectors
from ._text import to_native, to_bytes, to_text
from ansible.module_utils.common.text.converters import (
from ansible.module_utils.common.arg_spec import ModuleArgumentSpecValidator
from ansible.module_utils.common.text.formatters import (
import hashlib
from ansible.module_utils.six.moves.collections_abc import (
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.file import (
from ansible.module_utils.common.sys_info import (
from ansible.module_utils.pycompat24 import get_exception, literal_eval
from ansible.module_utils.common.parameters import (
from ansible.module_utils.errors import AnsibleFallbackNotFound, AnsibleValidationErrorMultiple, UnsupportedError
from ansible.module_utils.six import (
from ansible.module_utils.six.moves import map, reduce, shlex_quote
from ansible.module_utils.common.validation import (
from ansible.module_utils.common._utils import get_all_subclasses as _get_all_subclasses
from ansible.module_utils.parsing.convert_bool import BOOLEANS, BOOLEANS_FALSE, BOOLEANS_TRUE, boolean
from ansible.module_utils.common.warnings import (
@staticmethod
def _apply_operation_to_mode(user, operator, mode_to_apply, current_mode):
    if operator == '=':
        if user == 'u':
            mask = stat.S_IRWXU | stat.S_ISUID
        elif user == 'g':
            mask = stat.S_IRWXG | stat.S_ISGID
        elif user == 'o':
            mask = stat.S_IRWXO | stat.S_ISVTX
        inverse_mask = mask ^ PERM_BITS
        new_mode = current_mode & inverse_mask | mode_to_apply
    elif operator == '+':
        new_mode = current_mode | mode_to_apply
    elif operator == '-':
        new_mode = current_mode - (current_mode & mode_to_apply)
    return new_mode