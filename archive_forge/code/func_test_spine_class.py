import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.spines import Spines
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def test_spine_class():
    """Test Spines and SpinesProxy in isolation."""

    class SpineMock:

        def __init__(self):
            self.val = None

        def set(self, **kwargs):
            vars(self).update(kwargs)

        def set_val(self, val):
            self.val = val
    spines_dict = {'left': SpineMock(), 'right': SpineMock(), 'top': SpineMock(), 'bottom': SpineMock()}
    spines = Spines(**spines_dict)
    assert spines['left'] is spines_dict['left']
    assert spines.left is spines_dict['left']
    spines[['left', 'right']].set_val('x')
    assert spines.left.val == 'x'
    assert spines.right.val == 'x'
    assert spines.top.val is None
    assert spines.bottom.val is None
    spines[:].set_val('y')
    assert all((spine.val == 'y' for spine in spines.values()))
    spines[:].set(foo='bar')
    assert all((spine.foo == 'bar' for spine in spines.values()))
    with pytest.raises(AttributeError, match='foo'):
        spines.foo
    with pytest.raises(KeyError, match='foo'):
        spines['foo']
    with pytest.raises(KeyError, match='foo, bar'):
        spines[['left', 'foo', 'right', 'bar']]
    with pytest.raises(ValueError, match='single list'):
        spines['left', 'right']
    with pytest.raises(ValueError, match='Spines does not support slicing'):
        spines['left':'right']
    with pytest.raises(ValueError, match='Spines does not support slicing'):
        spines['top':]