import _thread
import codecs
import operator
import os
import pickle
import sys
import threading
from typing import Dict, TextIO
from _lsprof import Profiler, profiler_entry
from . import errors
def _entry(self, entry):
    out_file = self.out_file
    code = entry.code
    inlinetime = int(entry.inlinetime * 1000)
    if isinstance(code, str):
        out_file.write('fi=~\n')
    else:
        out_file.write('fi={}\n'.format(code.co_filename))
    out_file.write('fn={}\n'.format(label(code, True)))
    if isinstance(code, str):
        out_file.write('0  {}\n'.format(inlinetime))
    else:
        out_file.write('%d %d\n' % (code.co_firstlineno, inlinetime))
    if entry.calls:
        calls = entry.calls
    else:
        calls = []
    if isinstance(code, str):
        lineno = 0
    else:
        lineno = code.co_firstlineno
    for subentry in calls:
        self._subentry(lineno, subentry)
    out_file.write('\n')