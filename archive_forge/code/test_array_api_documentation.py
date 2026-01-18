import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
Same idea as `test_default_construction_fn_matrices`, but for the block
    creation function