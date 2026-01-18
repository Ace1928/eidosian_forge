import numpy as np
import ray
import ray.experimental.array.remote as ra
from . import core
@ray.remote(num_returns=2)
def tsqr_hr_helper1(u, s, y_top_block, b):
    y_top = y_top_block[:b, :b]
    s_full = np.diag(s)
    t = -1 * np.dot(u, np.dot(s_full, np.linalg.inv(y_top).T))
    return (t, y_top)