import contextlib
from io import StringIO
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from cycler import cycler
def test_property_collision_fill():
    fig, ax = plt.subplots()
    ax.set_prop_cycle(linewidth=[2, 3, 4, 5, 6], facecolor='bgcmy')
    t = range(10)
    for c in range(1, 4):
        ax.fill(t, t, lw=0.1)
    ax.fill(t, t)
    ax.fill(t, t)
    assert [p.get_facecolor() for p in ax.patches] == [mpl.colors.to_rgba(c) for c in 'bgcmy']
    assert [p.get_linewidth() for p in ax.patches] == [0.1, 0.1, 0.1, 5, 6]