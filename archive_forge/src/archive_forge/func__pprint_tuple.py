import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _pprint_tuple(self, object, stream, indent, allowance, context, level):
    stream.write('(')
    endchar = ',)' if len(object) == 1 else ')'
    self._format_items(object, stream, indent, allowance + len(endchar), context, level)
    stream.write(endchar)