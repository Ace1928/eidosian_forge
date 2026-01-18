import io
import sys
import time
import marshal
def trace_dispatch_c_call(self, frame, t):
    fn = ('', 0, self.c_func_name)
    self.cur = (t, 0, 0, fn, frame, self.cur)
    timings = self.timings
    if fn in timings:
        cc, ns, tt, ct, callers = timings[fn]
        timings[fn] = (cc, ns + 1, tt, ct, callers)
    else:
        timings[fn] = (0, 0, 0, 0, {})
    return 1