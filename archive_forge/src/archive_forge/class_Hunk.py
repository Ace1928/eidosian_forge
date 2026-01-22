import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
class Hunk:

    def __init__(self, orig_pos, orig_range, mod_pos, mod_range, tail=None):
        self.orig_pos = orig_pos
        self.orig_range = orig_range
        self.mod_pos = mod_pos
        self.mod_range = mod_range
        self.tail = tail
        self.lines = []

    def get_header(self):
        if self.tail is None:
            tail_str = b''
        else:
            tail_str = b' ' + self.tail
        return b'@@ -%s +%s @@%s\n' % (self.range_str(self.orig_pos, self.orig_range), self.range_str(self.mod_pos, self.mod_range), tail_str)

    def range_str(self, pos, range):
        """Return a file range, special-casing for 1-line files.

        :param pos: The position in the file
        :type pos: int
        :range: The range in the file
        :type range: int
        :return: a string in the format 1,4 except when range == pos == 1
        """
        if range == 1:
            return b'%i' % pos
        else:
            return b'%i,%i' % (pos, range)

    def as_bytes(self):
        lines = [self.get_header()]
        for line in self.lines:
            lines.append(line.as_bytes())
        return b''.join(lines)
    __bytes__ = as_bytes

    def shift_to_mod(self, pos):
        if pos < self.orig_pos - 1:
            return 0
        elif pos > self.orig_pos + self.orig_range:
            return self.mod_range - self.orig_range
        else:
            return self.shift_to_mod_lines(pos)

    def shift_to_mod_lines(self, pos):
        position = self.orig_pos - 1
        shift = 0
        for line in self.lines:
            if isinstance(line, InsertLine):
                shift += 1
            elif isinstance(line, RemoveLine):
                if position == pos:
                    return None
                shift -= 1
                position += 1
            elif isinstance(line, ContextLine):
                position += 1
            if position > pos:
                break
        return shift