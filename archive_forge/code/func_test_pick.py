import re
from matplotlib import path, transforms
from matplotlib.backend_bases import (
from matplotlib.backend_tools import RubberbandBase
from matplotlib.figure import Figure
from matplotlib.testing._markers import needs_pgf_xelatex
import matplotlib.pyplot as plt
import numpy as np
import pytest
def test_pick():
    fig = plt.figure()
    fig.text(0.5, 0.5, 'hello', ha='center', va='center', picker=True)
    fig.canvas.draw()
    picks = []

    def handle_pick(event):
        assert event.mouseevent.key == 'a'
        picks.append(event)
    fig.canvas.mpl_connect('pick_event', handle_pick)
    KeyEvent('key_press_event', fig.canvas, 'a')._process()
    MouseEvent('button_press_event', fig.canvas, *fig.transFigure.transform((0.5, 0.5)), MouseButton.LEFT)._process()
    KeyEvent('key_release_event', fig.canvas, 'a')._process()
    assert len(picks) == 1