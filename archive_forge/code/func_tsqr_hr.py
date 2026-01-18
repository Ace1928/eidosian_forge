import numpy as np
import ray
import ray.experimental.array.remote as ra
from . import core
@ray.remote(num_returns=4)
def tsqr_hr(a):
    q, r_temp = tsqr.remote(a)
    y, u, s = modified_lu.remote(q)
    y_blocked = ray.get(y)
    t, y_top = tsqr_hr_helper1.remote(u, s, y_blocked.object_refs[0, 0], a.shape[1])
    r = tsqr_hr_helper2.remote(s, r_temp)
    return (ray.get(y), ray.get(t), ray.get(y_top), ray.get(r))