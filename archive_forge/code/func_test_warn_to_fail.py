import warnings
import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.xfail(strict=True, reason='testing that warnings fail tests')
def test_warn_to_fail():
    warnings.warn('This should fail the test')