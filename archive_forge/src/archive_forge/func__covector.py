import numpy as np
from autograd.extend import VSpace
from autograd.builtins import NamedTupleVSpace
def _covector(self, x):
    return np.conj(x)