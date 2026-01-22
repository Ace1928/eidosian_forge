from __future__ import absolute_import
import cython
from io import StringIO
import re
import sys
from unicodedata import lookup as lookup_unicodechar, category as unicode_category
from functools import partial, reduce
from .Scanning import PyrexScanner, FileSourceDescriptor, tentatively_scan
from . import Nodes
from . import ExprNodes
from . import Builtin
from . import StringEncoding
from .StringEncoding import EncodedString, bytes_literal, _unicode, _bytes
from .ModuleNode import ModuleNode
from .Errors import error, warning
from .. import Utils
from . import Future
from . import Options
def _reject_cdef_modifier_in_py(s, name):
    """Step over incorrectly placed cdef modifiers (@see _CDEF_MODIFIERS) to provide a good error message for them.
    """
    if s.sy == 'IDENT' and name in _CDEF_MODIFIERS:
        s.error("Cannot use cdef modifier '%s' in Python function signature. Use a decorator instead." % name, fatal=False)
        return p_ident(s)
    return name