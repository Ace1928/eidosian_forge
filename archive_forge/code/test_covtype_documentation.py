from functools import partial
import pytest
from sklearn.datasets.tests.test_common import check_return_X_y
Test the covtype loader, if the data is available,
or if specifically requested via environment variable
(e.g. for CI jobs).