import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _pprint_list(self, object, stream, indent, allowance, context, level):
    stream.write('[')
    self._format_items(object, stream, indent, allowance + 1, context, level)
    stream.write(']')