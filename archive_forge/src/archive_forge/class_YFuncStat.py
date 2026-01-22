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
class YFuncStat(YStat):
    """
    Class holding information for function stats.
    """
    _KEYS = {'name': 0, 'module': 1, 'lineno': 2, 'ncall': 3, 'nactualcall': 4, 'builtin': 5, 'ttot': 6, 'tsub': 7, 'index': 8, 'children': 9, 'ctx_id': 10, 'ctx_name': 11, 'tag': 12, 'tavg': 14, 'full_name': 15}

    def __eq__(self, other):
        if other is None:
            return False
        return self.full_name == other.full_name

    def __ne__(self, other):
        return not self == other

    def __add__(self, other):
        if self is other:
            return self
        self.ncall += other.ncall
        self.nactualcall += other.nactualcall
        self.ttot += other.ttot
        self.tsub += other.tsub
        self.tavg = self.ttot / self.ncall
        for other_child_stat in other.children:
            self.children.append(other_child_stat)
        return self

    def __hash__(self):
        return hash(self.full_name)

    def is_recursive(self):
        if self.nactualcall == 0:
            return False
        return self.ncall != self.nactualcall

    def strip_dirs(self):
        self.module = os.path.basename(self.module)
        self.full_name = _func_fullname(self.builtin, self.module, self.lineno, self.name)
        return self

    def _print(self, out, columns):
        for x in sorted(columns.keys()):
            title, size = columns[x]
            if title == 'name':
                out.write(StatString(self.full_name).ltrim(size))
                out.write(' ' * COLUMN_GAP)
            elif title == 'ncall':
                if self.is_recursive():
                    out.write(StatString('%d/%d' % (self.ncall, self.nactualcall)).rtrim(size))
                else:
                    out.write(StatString(self.ncall).rtrim(size))
                out.write(' ' * COLUMN_GAP)
            elif title == 'tsub':
                out.write(StatString(_fft(self.tsub, size)).rtrim(size))
                out.write(' ' * COLUMN_GAP)
            elif title == 'ttot':
                out.write(StatString(_fft(self.ttot, size)).rtrim(size))
                out.write(' ' * COLUMN_GAP)
            elif title == 'tavg':
                out.write(StatString(_fft(self.tavg, size)).rtrim(size))
        out.write(LINESEP)