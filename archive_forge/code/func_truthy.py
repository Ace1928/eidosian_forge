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
def truthy(value, convert_bool=False):
    """Evaluate as value for truthiness using python ``bool``

    Optionally, attempt to do a conversion to bool from boolean like values
    such as ``"false"``, ``"true"``, ``"yes"``, ``"no"``, ``"on"``, ``"off"``, etc.

    .. versionadded:: 2.10
    """
    if convert_bool:
        try:
            value = boolean(value)
        except TypeError:
            pass
    return bool(value)