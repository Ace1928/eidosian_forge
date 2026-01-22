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
def _append_escape_sequence(kind, builder, escape_sequence, s):
    c = escape_sequence[1]
    if c in u'01234567':
        builder.append_charval(int(escape_sequence[1:], 8))
    elif c in u'\'"\\':
        builder.append(c)
    elif c in u'abfnrtv':
        builder.append(StringEncoding.char_from_escape_sequence(escape_sequence))
    elif c == u'\n':
        pass
    elif c == u'x':
        if len(escape_sequence) == 4:
            builder.append_charval(int(escape_sequence[2:], 16))
        else:
            s.error("Invalid hex escape '%s'" % escape_sequence, fatal=False)
    elif c in u'NUu' and kind in ('u', 'f', ''):
        chrval = -1
        if c == u'N':
            uchar = None
            try:
                uchar = lookup_unicodechar(escape_sequence[3:-1])
                chrval = ord(uchar)
            except KeyError:
                s.error('Unknown Unicode character name %s' % repr(escape_sequence[3:-1]).lstrip('u'), fatal=False)
            except TypeError:
                if uchar is not None and _IS_2BYTE_UNICODE and (len(uchar) == 2) and (unicode_category(uchar[0]) == 'Cs') and (unicode_category(uchar[1]) == 'Cs'):
                    chrval = 65536 + (ord(uchar[0]) - 55296) >> 10 + (ord(uchar[1]) - 56320)
                else:
                    raise
        elif len(escape_sequence) in (6, 10):
            chrval = int(escape_sequence[2:], 16)
            if chrval > 1114111:
                s.error("Invalid unicode escape '%s'" % escape_sequence)
                chrval = -1
        else:
            s.error("Invalid unicode escape '%s'" % escape_sequence, fatal=False)
        if chrval >= 0:
            builder.append_uescape(chrval, escape_sequence)
    else:
        builder.append(escape_sequence)