import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _pprint_ordered_dict(self, object, stream, indent, allowance, context, level):
    if not len(object):
        stream.write(repr(object))
        return
    cls = object.__class__
    stream.write(cls.__name__ + '(')
    self._format(list(object.items()), stream, indent + len(cls.__name__) + 1, allowance + 1, context, level)
    stream.write(')')