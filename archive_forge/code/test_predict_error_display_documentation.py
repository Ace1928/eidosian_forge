import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import load_diabetes
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.metrics import PredictionErrorDisplay
Check that we can tune the style of the lines.