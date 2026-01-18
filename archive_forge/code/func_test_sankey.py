import pytest
from numpy.testing import assert_allclose, assert_array_equal
from matplotlib.sankey import Sankey
from matplotlib.testing.decorators import check_figures_equal
def test_sankey():
    sankey = Sankey()
    sankey.add()