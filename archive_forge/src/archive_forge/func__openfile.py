import py
import sys
def _openfile(self):
    mode = self._append and 'a' or 'w'
    f = open(self._filename, mode)
    self._file = f