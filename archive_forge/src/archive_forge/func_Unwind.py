from eventlet.patcher import slurp_properties
import sys
import functools
from eventlet import greenthread
from eventlet import patcher
import _thread
def Unwind(self, cur, timings):
    """A function to unwind a 'cur' frame and tally the results"""
    'see profile.trace_dispatch_return() for details'
    while cur[-1]:
        rpt, rit, ret, rfn, frame, rcur = cur
        frame_total = rit + ret
        if rfn in timings:
            cc, ns, tt, ct, callers = timings[rfn]
        else:
            cc, ns, tt, ct, callers = (0, 0, 0, 0, {})
        if not ns:
            ct = ct + frame_total
            cc = cc + 1
        if rcur:
            ppt, pit, pet, pfn, pframe, pcur = rcur
        else:
            pfn = None
        if pfn in callers:
            callers[pfn] = callers[pfn] + 1
        elif pfn:
            callers[pfn] = 1
        timings[rfn] = (cc, ns - 1, tt + rit, ct, callers)
        ppt, pit, pet, pfn, pframe, pcur = rcur
        rcur = (ppt, pit + rpt, pet + frame_total, pfn, pframe, pcur)
        cur = rcur
    return cur