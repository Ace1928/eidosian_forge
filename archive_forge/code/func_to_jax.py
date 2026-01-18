import numpy as np
from ..sharing import to_backend_cache_wrap
@to_backend_cache_wrap
@jax.jit
def to_jax(x):
    return x