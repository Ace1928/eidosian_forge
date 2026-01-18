import warnings
import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal
def test_wrap_failure():
    with pytest.raises(ValueError, match='^The decorated function'):

        @check_figures_equal()
        def should_fail(test, ref):
            pass