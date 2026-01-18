import io
import sys
import time
import marshal
def trace_dispatch_return(self, frame, t):
    if frame is not self.cur[-2]:
        assert frame is self.cur[-2].f_back, ('Bad return', self.cur[-3])
        self.trace_dispatch_return(self.cur[-2], 0)
    rpt, rit, ret, rfn, frame, rcur = self.cur
    rit = rit + t
    frame_total = rit + ret
    ppt, pit, pet, pfn, pframe, pcur = rcur
    self.cur = (ppt, pit + rpt, pet + frame_total, pfn, pframe, pcur)
    timings = self.timings
    cc, ns, tt, ct, callers = timings[rfn]
    if not ns:
        ct = ct + frame_total
        cc = cc + 1
    if pfn in callers:
        callers[pfn] = callers[pfn] + 1
    else:
        callers[pfn] = 1
    timings[rfn] = (cc, ns - 1, tt + rit, ct, callers)
    return 1