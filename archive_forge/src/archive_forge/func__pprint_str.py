import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _pprint_str(self, object, stream, indent, allowance, context, level):
    write = stream.write
    if not len(object):
        write(repr(object))
        return
    chunks = []
    lines = object.splitlines(True)
    if level == 1:
        indent += 1
        allowance += 1
    max_width1 = max_width = self._width - indent
    for i, line in enumerate(lines):
        rep = repr(line)
        if i == len(lines) - 1:
            max_width1 -= allowance
        if len(rep) <= max_width1:
            chunks.append(rep)
        else:
            parts = re.findall('\\S*\\s*', line)
            assert parts
            assert not parts[-1]
            parts.pop()
            max_width2 = max_width
            current = ''
            for j, part in enumerate(parts):
                candidate = current + part
                if j == len(parts) - 1 and i == len(lines) - 1:
                    max_width2 -= allowance
                if len(repr(candidate)) > max_width2:
                    if current:
                        chunks.append(repr(current))
                    current = part
                else:
                    current = candidate
            if current:
                chunks.append(repr(current))
    if len(chunks) == 1:
        write(rep)
        return
    if level == 1:
        write('(')
    for i, rep in enumerate(chunks):
        if i > 0:
            write('\n' + ' ' * indent)
        write(rep)
    if level == 1:
        write(')')