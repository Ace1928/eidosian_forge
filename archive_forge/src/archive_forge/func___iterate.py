from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
import collections
import warnings
def __iterate(self, stack):
    while stack:
        path = stack.popleft()
        yield path
        try:
            subkeys = self.subkeys(path)
        except WindowsError:
            continue
        prefix = path + '\\'
        subkeys = [prefix + name for name in subkeys]
        stack.extendleft(subkeys)