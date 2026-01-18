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
def xyzlim(vmin, vmax=None):
    """Set limits or all axis the same, if vmax not given, use [-vmin, vmin]."""
    if vmax is None:
        vmin, vmax = (-vmin, vmin)
    xlim(vmin, vmax)
    ylim(vmin, vmax)
    zlim(vmin, vmax)