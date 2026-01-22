import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import weakref
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.testing.decorators import check_figures_equal
@pytest.fixture()
def anim(request):
    """Create a simple animation (with options)."""
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        x = np.linspace(0, 10, 100)
        y = np.sin(x + i)
        line.set_data(x, y)
        return (line,)
    kwargs = dict(getattr(request, 'param', {}))
    klass = kwargs.pop('klass', animation.FuncAnimation)
    if 'frames' not in kwargs:
        kwargs['frames'] = 5
    return klass(fig=fig, func=animate, init_func=init, **kwargs)