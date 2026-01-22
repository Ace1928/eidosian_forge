from __future__ import absolute_import
from __future__ import division
import pythreejs
import os
import time
import warnings
import tempfile
import uuid
import base64
from io import BytesIO as StringIO
import six
import numpy as np
import PIL.Image
import matplotlib.style
import ipywidgets
import IPython
from IPython.display import display
import ipyvolume as ipv
import ipyvolume.embed
from ipyvolume import utils
from . import ui
def _grow_limits(x, y, z):
    fig = gcf()
    xlim(*_grow_limit(fig.xlim, x))
    ylim(*_grow_limit(fig.ylim, y))
    zlim(*_grow_limit(fig.zlim, z))