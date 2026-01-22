from eventlet.patcher import slurp_properties
import sys
import functools
from eventlet import greenthread
from eventlet import patcher
import _thread
def TallyTimings(self):
    oldtimings = self.sleeping
    self.sleeping = {}
    self.cur = self.Unwind(self.cur, self.timings)
    for tasklet, (cur, timings) in oldtimings.items():
        self.Unwind(cur, timings)
        for k, v in timings.items():
            if k not in self.timings:
                self.timings[k] = v
            else:
                cc, ns, tt, ct, callers = self.timings[k]
                cc += v[0]
                tt += v[2]
                ct += v[3]
                for k1, v1 in v[4].items():
                    callers[k1] = callers.get(k1, 0) + v1
                self.timings[k] = (cc, ns, tt, ct, callers)