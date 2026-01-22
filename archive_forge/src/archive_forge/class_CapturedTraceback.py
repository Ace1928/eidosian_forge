from types import TracebackType
from typing import List, Optional
import tempfile
import traceback
import contextlib
import inspect
import os.path
class CapturedTraceback:
    __slots__ = ['tb', 'skip']

    def __init__(self, tb, skip=0):
        self.tb = tb
        self.skip = skip

    def cleanup(self):
        self.tb = None

    def summary(self):
        import torch._C._profiler
        if self.tb is None:
            return traceback.StackSummary()
        return _extract_symbolized_tb(torch._C._profiler.symbolize_tracebacks([self.tb])[0], self.skip)

    def __getstate__(self):
        return (None, {'tb': None, 'skip': self.skip})

    @staticmethod
    def extract(*, script=False, cpp=False, skip=0):
        """
        Like traceback.extract_stack(), but faster (approximately 20x faster); it
        is fast enough that you can unconditionally log stacks this way as part of
        normal execution.  It returns a torch._C._profiler.CapturedTraceback
        object that must be formatted specially with format_captured_tb.

        By default, this only reports Python backtraces (like extract_stack).  You
        can set the script/cpp kwargs to also turn on TorchScript/C++ trace
        reporting.
        """
        import torch._C._profiler
        if script or cpp:
            assert skip == 0, 'skip with script/cpp NYI'
        return CapturedTraceback(torch._C._profiler.gather_traceback(python=True, script=script, cpp=cpp), 0 if script or cpp else skip + 1)

    def format(self):
        """
        Formats a single torch._C._profiler.CapturedTraceback into a list of
        strings equivalent to the output of traceback.format_list.  Note that if
        pass it CapturedTraceback with C++ traces,  it is better not to use this
        function and use the batch formatting API format_captured_tbs to amortize
        the cost of symbolization
        """
        return traceback.format_list(self.summary())

    @staticmethod
    def format_all(tbs):
        """
        Bulk version of CapturedTraceback.format.  Returns a list of list of strings.
        """
        import torch._C._profiler
        rs: List[Optional[List[str]]] = []
        delayed_idxs = []
        for i, tb in enumerate(tbs):
            if tb.tb is None:
                rs.append([])
            else:
                rs.append(None)
                delayed_idxs.append(i)
        stbs = torch._C._profiler.symbolize_tracebacks([tbs[i].tb for i in delayed_idxs])
        for i, stb in zip(delayed_idxs, stbs):
            rs[i] = traceback.format_list(tbs[i].summary())
        return rs