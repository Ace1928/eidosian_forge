import warnings
import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('a', [1])
@check_figures_equal(extensions=['png'])
@pytest.mark.parametrize('b', [1])
def test_parametrize_with_check_figure_equal(a, fig_ref, b, fig_test):
    assert a == b