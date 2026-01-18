import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
def test_suplabels():
    fig, ax = plt.subplots(layout='constrained')
    fig.draw_without_rendering()
    pos0 = ax.get_tightbbox(fig.canvas.get_renderer())
    fig.supxlabel('Boo')
    fig.supylabel('Booy')
    fig.draw_without_rendering()
    pos = ax.get_tightbbox(fig.canvas.get_renderer())
    assert pos.y0 > pos0.y0 + 10.0
    assert pos.x0 > pos0.x0 + 10.0
    fig, ax = plt.subplots(layout='constrained')
    fig.draw_without_rendering()
    pos0 = ax.get_tightbbox(fig.canvas.get_renderer())
    fig.supxlabel('Boo', x=0.5)
    fig.supylabel('Boo', y=0.5)
    fig.draw_without_rendering()
    pos = ax.get_tightbbox(fig.canvas.get_renderer())
    assert pos.y0 > pos0.y0 + 10.0
    assert pos.x0 > pos0.x0 + 10.0