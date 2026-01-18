import copy
from datetime import datetime
import io
from pathlib import Path
import pickle
import platform
from threading import Timer
from types import SimpleNamespace
import warnings
import numpy as np
import pytest
from PIL import Image
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure, FigureBase
from matplotlib.layout_engine import (ConstrainedLayoutEngine,
from matplotlib.ticker import AutoMinorLocator, FixedFormatter, ScalarFormatter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
@image_comparison(['test_subfigure_double.png'], style='mpl20', savefig_kwarg={'facecolor': 'teal'})
def test_subfigure_double():
    np.random.seed(19680801)
    fig = plt.figure(layout='constrained', figsize=(10, 8))
    fig.suptitle('fig')
    subfigs = fig.subfigures(1, 2, wspace=0.07)
    subfigs[0].set_facecolor('coral')
    subfigs[0].suptitle('subfigs[0]')
    subfigs[1].set_facecolor('coral')
    subfigs[1].suptitle('subfigs[1]')
    subfigsnest = subfigs[0].subfigures(2, 1, height_ratios=[1, 1.4])
    subfigsnest[0].suptitle('subfigsnest[0]')
    subfigsnest[0].set_facecolor('r')
    axsnest0 = subfigsnest[0].subplots(1, 2, sharey=True)
    for ax in axsnest0:
        fontsize = 12
        pc = ax.pcolormesh(np.random.randn(30, 30), vmin=-2.5, vmax=2.5)
        ax.set_xlabel('x-label', fontsize=fontsize)
        ax.set_ylabel('y-label', fontsize=fontsize)
        ax.set_title('Title', fontsize=fontsize)
    subfigsnest[0].colorbar(pc, ax=axsnest0)
    subfigsnest[1].suptitle('subfigsnest[1]')
    subfigsnest[1].set_facecolor('g')
    axsnest1 = subfigsnest[1].subplots(3, 1, sharex=True)
    for nn, ax in enumerate(axsnest1):
        ax.set_ylabel(f'ylabel{nn}')
    subfigsnest[1].supxlabel('supxlabel')
    subfigsnest[1].supylabel('supylabel')
    axsRight = subfigs[1].subplots(2, 2)