import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _pprint_bytes(self, object, stream, indent, allowance, context, level):
    write = stream.write
    if len(object) <= 4:
        write(repr(object))
        return
    parens = level == 1
    if parens:
        indent += 1
        allowance += 1
        write('(')
    delim = ''
    for rep in _wrap_bytes_repr(object, self._width - indent, allowance):
        write(delim)
        write(rep)
        if not delim:
            delim = '\n' + ' ' * indent
    if parens:
        write(')')