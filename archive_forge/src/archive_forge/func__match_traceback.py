from collections.abc import Sequence, Iterable
from functools import total_ordering
import fnmatch
import linecache
import os.path
import pickle
from _tracemalloc import *
from _tracemalloc import _get_object_traceback, _get_traces
def _match_traceback(self, traceback):
    if self.all_frames:
        if any((self._match_frame_impl(filename, lineno) for filename, lineno in traceback)):
            return self.inclusive
        else:
            return not self.inclusive
    else:
        filename, lineno = traceback[0]
        return self._match_frame(filename, lineno)