import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _format_items(self, items, stream, indent, allowance, context, level):
    write = stream.write
    indent += self._indent_per_level
    if self._indent_per_level > 1:
        write((self._indent_per_level - 1) * ' ')
    delimnl = ',\n' + ' ' * indent
    delim = ''
    width = max_width = self._width - indent + 1
    it = iter(items)
    try:
        next_ent = next(it)
    except StopIteration:
        return
    last = False
    while not last:
        ent = next_ent
        try:
            next_ent = next(it)
        except StopIteration:
            last = True
            max_width -= allowance
            width -= allowance
        if self._compact:
            rep = self._repr(ent, context, level)
            w = len(rep) + 2
            if width < w:
                width = max_width
                if delim:
                    delim = delimnl
            if width >= w:
                width -= w
                write(delim)
                delim = ', '
                write(rep)
                continue
        write(delim)
        delim = delimnl
        self._format(ent, stream, indent, allowance if last else 1, context, level)