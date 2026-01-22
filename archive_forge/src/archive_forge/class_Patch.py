import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
class Patch(BinaryPatch):

    def __init__(self, oldname, newname, oldts=None, newts=None):
        BinaryPatch.__init__(self, oldname, newname)
        self.oldts = oldts
        self.newts = newts
        self.hunks = []

    def as_bytes(self):
        ret = self.get_header()
        ret += b''.join([h.as_bytes() for h in self.hunks])
        return ret

    @classmethod
    def _headerline(cls, start, name, ts):
        l = start + b' ' + name
        if ts is not None:
            l += b'\t%s' % ts
        l += b'\n'
        return l

    def get_header(self):
        return self._headerline(b'---', self.oldname, self.oldts) + self._headerline(b'+++', self.newname, self.newts)

    def stats_values(self):
        """Calculate the number of inserts and removes."""
        removes = 0
        inserts = 0
        for hunk in self.hunks:
            for line in hunk.lines:
                if isinstance(line, InsertLine):
                    inserts += 1
                elif isinstance(line, RemoveLine):
                    removes += 1
        return (inserts, removes, len(self.hunks))

    def stats_str(self):
        """Return a string of patch statistics"""
        return '%i inserts, %i removes in %i hunks' % self.stats_values()

    def pos_in_mod(self, position):
        newpos = position
        for hunk in self.hunks:
            shift = hunk.shift_to_mod(position)
            if shift is None:
                return None
            newpos += shift
        return newpos

    def iter_inserted(self):
        """Iteraties through inserted lines

        :return: Pair of line number, line
        :rtype: iterator of (int, InsertLine)
        """
        for hunk in self.hunks:
            pos = hunk.mod_pos - 1
            for line in hunk.lines:
                if isinstance(line, InsertLine):
                    yield (pos, line)
                    pos += 1
                if isinstance(line, ContextLine):
                    pos += 1