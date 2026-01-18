import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
Make sure bin mapper treats negative categories as missing values.