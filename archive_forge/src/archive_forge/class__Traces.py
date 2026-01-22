from collections.abc import Sequence, Iterable
from functools import total_ordering
import fnmatch
import linecache
import os.path
import pickle
from _tracemalloc import *
from _tracemalloc import _get_object_traceback, _get_traces
class _Traces(Sequence):

    def __init__(self, traces):
        Sequence.__init__(self)
        self._traces = traces

    def __len__(self):
        return len(self._traces)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple((Trace(trace) for trace in self._traces[index]))
        else:
            return Trace(self._traces[index])

    def __contains__(self, trace):
        return trace._trace in self._traces

    def __eq__(self, other):
        if not isinstance(other, _Traces):
            return NotImplemented
        return self._traces == other._traces

    def __repr__(self):
        return '<Traces len=%s>' % len(self)