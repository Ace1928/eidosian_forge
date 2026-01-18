from contextlib import ExitStack
from copy import copy
import functools
import io
import os
from pathlib import Path
import platform
import sys
import urllib.request
import numpy as np
from numpy.testing import assert_array_equal
from PIL import Image
import matplotlib as mpl
from matplotlib import (
from matplotlib.image import (AxesImage, BboxImage, FigureImage,
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.transforms import Bbox, Affine2D, TransformedBbox
import matplotlib.ticker as mticker
import pytest
def test_respects_bbox():
    fig, axs = plt.subplots(2)
    for ax in axs:
        ax.set_axis_off()
    im = axs[1].imshow([[0, 1], [2, 3]], aspect='auto', extent=(0, 1, 0, 1))
    im.set_clip_path(None)
    im.set_clip_box(axs[0].bbox)
    buf_before = io.BytesIO()
    fig.savefig(buf_before, format='rgba')
    assert {*buf_before.getvalue()} == {255}
    axs[1].set(ylim=(-1, 0))
    buf_after = io.BytesIO()
    fig.savefig(buf_after, format='rgba')
    assert buf_before.getvalue() != buf_after.getvalue()