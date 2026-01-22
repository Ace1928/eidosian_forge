import numpy as np
def _is_out(ind, shape):
    for i, s in zip(ind, shape):
        if i < 0:
            return True
        if i >= s:
            return True
    return False