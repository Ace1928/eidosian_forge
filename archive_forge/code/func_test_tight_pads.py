import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
def test_tight_pads():
    fig, ax = plt.subplots()
    with pytest.warns(PendingDeprecationWarning, match='will be deprecated'):
        fig.set_tight_layout({'pad': 0.15})
    fig.draw_without_rendering()