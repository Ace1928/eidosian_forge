import copy
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from matplotlib import scale
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_transform_shortcuts(self):
    assert self.stack1 - self.stack2_subset == self.ta1
    assert self.stack2 - self.stack2_subset == self.ta1
    assert self.stack2_subset - self.stack2 == self.ta1.inverted()
    assert (self.stack2_subset - self.stack2).depth == 1
    with pytest.raises(ValueError):
        self.stack1 - self.stack2
    aff1 = self.ta1 + (self.ta2 + self.ta3)
    aff2 = self.ta2 + self.ta3
    assert aff1 - aff2 == self.ta1
    assert aff1 - self.ta2 == aff1 + self.ta2.inverted()
    assert self.stack1 - self.ta3 == self.ta1 + (self.tn1 + self.ta2)
    assert self.stack2 - self.ta3 == self.ta1 + self.tn1 + self.ta2
    assert self.ta2 + self.ta3 - self.ta3 + self.ta3 == self.ta2 + self.ta3