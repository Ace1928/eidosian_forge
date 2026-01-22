import importlib
import importlib.util
import inspect
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from PIL import Image
import pytest
import matplotlib as mpl
from matplotlib import _c_internal_utils
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.testing import subprocess_run_helper as _run_helper
def _test_number_of_draws_script():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ln, = ax.plot([0, 1], [1, 2], animated=True)
    plt.show(block=False)
    plt.pause(0.3)
    fig.canvas.mpl_connect('draw_event', print)
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    ax.draw_artist(ln)
    fig.canvas.blit(fig.bbox)
    for j in range(10):
        fig.canvas.restore_region(bg)
        ln, = ax.plot([0, 1], [1, 2])
        ax.draw_artist(ln)
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()
    plt.pause(0.1)