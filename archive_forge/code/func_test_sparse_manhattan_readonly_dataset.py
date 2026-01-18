import warnings
from types import GeneratorType
import numpy as np
from numpy import linalg
from scipy.sparse import issparse
from scipy.spatial.distance import (
import pytest
from sklearn import config_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.parallel import Parallel, delayed
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_sparse_manhattan_readonly_dataset(csr_container):
    matrices1 = [csr_container(np.ones((5, 5)))]
    matrices2 = [csr_container(np.ones((5, 5)))]
    Parallel(n_jobs=2, max_nbytes=0)((delayed(manhattan_distances)(m1, m2) for m1, m2 in zip(matrices1, matrices2)))