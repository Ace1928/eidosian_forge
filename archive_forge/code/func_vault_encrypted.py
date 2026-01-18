from __future__ import (absolute_import, division, print_function)
import re
import operator as py_operator
from collections.abc import MutableMapping, MutableSequence
from ansible.module_utils.compat.version import LooseVersion, StrictVersion
from ansible import errors
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.utils.display import Display
from ansible.utils.version import SemanticVersion
def vault_encrypted(value):
    """Evaulate whether a variable is a single vault encrypted value

    .. versionadded:: 2.10
    """
    return getattr(value, '__ENCRYPTED__', False) and value.is_encrypted()