from __future__ import (absolute_import, division, print_function)
import keyword
import random
import uuid
from collections.abc import MutableMapping, MutableSequence
from json import dumps
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.parsing.splitter import parse_kv
def _isidentifier_PY3(ident):
    if not isinstance(ident, string_types):
        return False
    if not ident.isascii():
        return False
    if not ident.isidentifier():
        return False
    if keyword.iskeyword(ident):
        return False
    return True