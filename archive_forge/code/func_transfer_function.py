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
def transfer_function(level=[0.1, 0.5, 0.9], opacity=[0.01, 0.05, 0.1], level_width=0.1, controls=True, max_opacity=0.2):
    """Create a transfer function, see volshow."""
    tf_kwargs = {}
    try:
        level[0]
    except:
        level = [level]
    try:
        opacity[0]
    except:
        opacity = [opacity] * 3
    try:
        level_width[0]
    except:
        level_width = [level_width] * 3
    min_length = min(len(level), len(level_width), len(opacity))
    level = list(level[:min_length])
    opacity = list(opacity[:min_length])
    level_width = list(level_width[:min_length])
    while len(level) < 3:
        level.append(0)
    while len(opacity) < 3:
        opacity.append(0)
    while len(level_width) < 3:
        level_width.append(0)
    for i in range(1, 4):
        tf_kwargs['level' + str(i)] = level[i - 1]
        tf_kwargs['opacity' + str(i)] = opacity[i - 1]
        tf_kwargs['width' + str(i)] = level_width[i - 1]
    tf = ipv.TransferFunctionWidgetJs3(**tf_kwargs)
    gcf()
    if controls:
        current.container.children = [tf.control(max_opacity=max_opacity)] + current.container.children
    return tf