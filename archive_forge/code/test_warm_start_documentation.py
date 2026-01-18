import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
from sklearn.metrics import check_scoring
Assert that two HistGBM instances are identical.