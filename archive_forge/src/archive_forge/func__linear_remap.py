import math
def _linear_remap(v, src_interval, dst_interval):
    l0, r0 = src_interval
    l1, r1 = dst_interval
    return l1 + (r1 - l1) / (r0 - l0) * (v - l0)