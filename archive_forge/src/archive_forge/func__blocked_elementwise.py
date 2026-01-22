import numpy as np
from scipy.sparse.linalg import aslinearoperator
def _blocked_elementwise(func):
    """
    Decorator for an elementwise function, to apply it blockwise along
    first dimension, to avoid excessive memory usage in temporaries.
    """
    block_size = 2 ** 20

    def wrapper(x):
        if x.shape[0] < block_size:
            return func(x)
        else:
            y0 = func(x[:block_size])
            y = np.zeros((x.shape[0],) + y0.shape[1:], dtype=y0.dtype)
            y[:block_size] = y0
            del y0
            for j in range(block_size, x.shape[0], block_size):
                y[j:j + block_size] = func(x[j:j + block_size])
            return y
    return wrapper