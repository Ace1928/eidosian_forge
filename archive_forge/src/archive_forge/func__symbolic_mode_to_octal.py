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
@classmethod
def _symbolic_mode_to_octal(cls, path_stat, symbolic_mode):
    """
        This enables symbolic chmod string parsing as stated in the chmod man-page

        This includes things like: "u=rw-x+X,g=r-x+X,o=r-x+X"
        """
    new_mode = stat.S_IMODE(path_stat.st_mode)
    for mode in symbolic_mode.split(','):
        permlist = MODE_OPERATOR_RE.split(mode)
        opers = MODE_OPERATOR_RE.findall(mode)
        users = permlist.pop(0)
        use_umask = users == ''
        if users == 'a' or users == '':
            users = 'ugo'
        if not USERS_RE.match(users):
            raise ValueError('bad symbolic permission for mode: %s' % mode)
        for idx, perms in enumerate(permlist):
            if not PERMS_RE.match(perms):
                raise ValueError('bad symbolic permission for mode: %s' % mode)
            for user in users:
                mode_to_apply = cls._get_octal_mode_from_symbolic_perms(path_stat, user, perms, use_umask, new_mode)
                new_mode = cls._apply_operation_to_mode(user, opers[idx], mode_to_apply, new_mode)
    return new_mode