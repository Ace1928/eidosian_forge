import itertools
import numpy as np
import pytest
from matplotlib.axes import Axes, SubplotBase
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
@pytest.mark.parametrize('width_ratios', [None, [1, 3, 2]])
@pytest.mark.parametrize('height_ratios', [None, [1, 2]])
@check_figures_equal(extensions=['png'])
def test_width_and_height_ratios(fig_test, fig_ref, height_ratios, width_ratios):
    fig_test.subplots(2, 3, height_ratios=height_ratios, width_ratios=width_ratios)
    fig_ref.subplots(2, 3, gridspec_kw={'height_ratios': height_ratios, 'width_ratios': width_ratios})