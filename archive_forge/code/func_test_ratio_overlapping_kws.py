import itertools
import numpy as np
import pytest
from matplotlib.axes import Axes, SubplotBase
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
@pytest.mark.parametrize('method,args', [('subplots', (2, 3)), ('subplot_mosaic', ('abc;def',))])
def test_ratio_overlapping_kws(method, args):
    with pytest.raises(ValueError, match='height_ratios'):
        getattr(plt, method)(*args, height_ratios=[1, 2], gridspec_kw={'height_ratios': [1, 2]})
    with pytest.raises(ValueError, match='width_ratios'):
        getattr(plt, method)(*args, width_ratios=[1, 2, 3], gridspec_kw={'width_ratios': [1, 2, 3]})