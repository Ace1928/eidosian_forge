import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
def test_tight_toggle():
    fig, ax = plt.subplots()
    with pytest.warns(PendingDeprecationWarning):
        fig.set_tight_layout(True)
        assert fig.get_tight_layout()
        fig.set_tight_layout(False)
        assert not fig.get_tight_layout()
        fig.set_tight_layout(True)
        assert fig.get_tight_layout()