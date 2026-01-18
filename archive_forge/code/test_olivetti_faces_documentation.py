import numpy as np
from sklearn.datasets.tests.test_common import check_return_X_y
from sklearn.utils import Bunch
from sklearn.utils._testing import assert_array_equal
Test Olivetti faces fetcher, if the data is available,
or if specifically requested via environment variable
(e.g. for CI jobs).