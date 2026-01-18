import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.cluster import FeatureAgglomeration
from sklearn.datasets import make_blobs
from sklearn.utils._testing import assert_array_almost_equal
Check `get_feature_names_out` for `FeatureAgglomeration`.