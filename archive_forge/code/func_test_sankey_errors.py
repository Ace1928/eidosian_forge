import pytest
from numpy.testing import assert_allclose, assert_array_equal
from matplotlib.sankey import Sankey
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('kwargs, msg', (({'gap': -1}, "'gap' is negative"), ({'gap': 1, 'radius': 2}, "'radius' is greater than 'gap'"), ({'head_angle': -1}, "'head_angle' is negative"), ({'tolerance': -1}, "'tolerance' is negative"), ({'flows': [1, -1], 'orientations': [-1, 0, 1]}, "The shapes of 'flows' \\(2,\\) and 'orientations'"), ({'flows': [1, -1], 'labels': ['a', 'b', 'c']}, "The shapes of 'flows' \\(2,\\) and 'labels'")))
def test_sankey_errors(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        Sankey(**kwargs)