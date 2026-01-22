import io
import sys
import time
import marshal
def _calibrate_inner(self, m, verbose):
    get_time = self.get_time

    def f1(n):
        for i in range(n):
            x = 1

    def f(m, f1=f1):
        for i in range(m):
            f1(100)
    f(m)
    t0 = get_time()
    f(m)
    t1 = get_time()
    elapsed_noprofile = t1 - t0
    if verbose:
        print('elapsed time without profiling =', elapsed_noprofile)
    p = Profile()
    t0 = get_time()
    p.runctx('f(m)', globals(), locals())
    t1 = get_time()
    elapsed_profile = t1 - t0
    if verbose:
        print('elapsed time with profiling =', elapsed_profile)
    total_calls = 0.0
    reported_time = 0.0
    for (filename, line, funcname), (cc, ns, tt, ct, callers) in p.timings.items():
        if funcname in ('f', 'f1'):
            total_calls += cc
            reported_time += tt
    if verbose:
        print("'CPU seconds' profiler reported =", reported_time)
        print('total # calls =', total_calls)
    if total_calls != m + 1:
        raise ValueError('internal error: total calls = %d' % total_calls)
    mean = (reported_time - elapsed_noprofile) / 2.0 / total_calls
    if verbose:
        print('mean stopwatch overhead per profile event =', mean)
    return mean