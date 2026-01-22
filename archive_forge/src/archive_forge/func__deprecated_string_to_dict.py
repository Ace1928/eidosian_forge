from __future__ import annotations
import datetime
import functools
import json
import re
import shlex
import typing as t
from tokenize import COMMENT, TokenInfo
import astroid
from pylint.checkers import BaseChecker, BaseTokenChecker
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.six import string_types
from ansible.release import __version__ as ansible_version_raw
from ansible.utils.version import SemanticVersion
def _deprecated_string_to_dict(self, token: TokenInfo, string: str) -> dict[str, str]:
    valid_keys = {'description', 'core_version', 'python_version'}
    data = dict.fromkeys(valid_keys)
    for opt in shlex.split(string):
        if '=' not in opt:
            data[opt] = None
            continue
        key, _sep, value = opt.partition('=')
        data[key] = value
    if not any((data['core_version'], data['python_version'])):
        self.add_message('ansible-deprecated-version-comment-missing-version', line=token.start[0], col_offset=token.start[1])
    bad = set(data).difference(valid_keys)
    if bad:
        self.add_message('ansible-deprecated-version-comment-invalid-key', line=token.start[0], col_offset=token.start[1], args=(','.join(bad),))
    return data