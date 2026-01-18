from eventlet.patcher import slurp_properties
import sys
import functools
from eventlet import greenthread
from eventlet import patcher
import _thread
def trace_dispatch_c_return_extend_back(self, frame, t):
    if isinstance(self.cur[-2], Profile.fake_frame):
        return False
        self.trace_dispatch_c_call(frame, 0)
    return self.trace_dispatch_return(frame, t)