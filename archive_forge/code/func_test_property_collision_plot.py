import contextlib
from io import StringIO
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from cycler import cycler
def test_property_collision_plot():
    fig, ax = plt.subplots()
    ax.set_prop_cycle('linewidth', [2, 4])
    t = range(10)
    for c in range(1, 4):
        ax.plot(t, t, lw=0.1)
    ax.plot(t, t)
    ax.plot(t, t)
    assert [l.get_linewidth() for l in ax.lines] == [0.1, 0.1, 0.1, 2, 4]