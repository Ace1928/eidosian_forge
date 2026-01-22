import numpy as np
from autograd.extend import VSpace
from autograd.builtins import NamedTupleVSpace
class SVDResultVSpace(NamedTupleVSpace):
    seq_type = np.linalg.linalg.SVDResult