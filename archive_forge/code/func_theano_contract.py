import numpy as np
from ..sharing import to_backend_cache_wrap
def theano_contract(*arrays):
    return graph(*[x for x in arrays if not isinstance(x, theano.tensor.TensorConstant)])