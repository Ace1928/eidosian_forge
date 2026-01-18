from collections import namedtuple
import io
import numpy as np
from numpy.testing import assert_allclose
import pytest
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.offsetbox import (
@check_figures_equal()
def test_textarea_set_text(fig_test, fig_ref):
    ax_ref = fig_ref.add_subplot()
    text0 = AnchoredText('Foo', 'upper left')
    ax_ref.add_artist(text0)
    ax_test = fig_test.add_subplot()
    text1 = AnchoredText('Bar', 'upper left')
    ax_test.add_artist(text1)
    text1.txt.set_text('Foo')