import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
class YThreadStat(YStat):
    """
    Class holding information for thread stats.
    """
    _KEYS = {'name': 0, 'id': 1, 'tid': 2, 'ttot': 3, 'sched_count': 4}

    def __eq__(self, other):
        if other is None:
            return False
        return self.id == other.id

    def __ne__(self, other):
        return not self == other

    def __hash__(self, *args, **kwargs):
        return hash(self.id)

    def _print(self, out, columns):
        for x in sorted(columns.keys()):
            title, size = columns[x]
            if title == 'name':
                out.write(StatString(self.name).ltrim(size))
                out.write(' ' * COLUMN_GAP)
            elif title == 'id':
                out.write(StatString(self.id).rtrim(size))
                out.write(' ' * COLUMN_GAP)
            elif title == 'tid':
                out.write(StatString(self.tid).rtrim(size))
                out.write(' ' * COLUMN_GAP)
            elif title == 'ttot':
                out.write(StatString(_fft(self.ttot, size)).rtrim(size))
                out.write(' ' * COLUMN_GAP)
            elif title == 'scnt':
                out.write(StatString(self.sched_count).rtrim(size))
        out.write(LINESEP)