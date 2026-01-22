import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _pprint_bytearray(self, object, stream, indent, allowance, context, level):
    write = stream.write
    write('bytearray(')
    self._pprint_bytes(bytes(object), stream, indent + 10, allowance + 1, context, level + 1)
    write(')')