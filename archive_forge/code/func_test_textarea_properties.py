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
def test_textarea_properties():
    ta = TextArea('Foo')
    assert ta.get_text() == 'Foo'
    assert not ta.get_multilinebaseline()
    ta.set_text('Bar')
    ta.set_multilinebaseline(True)
    assert ta.get_text() == 'Bar'
    assert ta.get_multilinebaseline()