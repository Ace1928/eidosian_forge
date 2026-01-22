import math
import os
import logging
from pathlib import Path
import warnings
import numpy as np
import PIL.Image
import PIL.PngImagePlugin
import matplotlib as mpl
from matplotlib import _api, cbook, cm
from matplotlib import _image
from matplotlib._image import *
import matplotlib.artist as martist
from matplotlib.backend_bases import FigureCanvasBase
import matplotlib.colors as mcolors
from matplotlib.transforms import (
def _pil_png_to_float_array(pil_png):
    """Convert a PIL `PNGImageFile` to a 0-1 float array."""
    mode = pil_png.mode
    rawmode = pil_png.png.im_rawmode
    if rawmode == '1':
        return np.asarray(pil_png, np.float32)
    if rawmode == 'L;2':
        return np.divide(pil_png, 2 ** 2 - 1, dtype=np.float32)
    if rawmode == 'L;4':
        return np.divide(pil_png, 2 ** 4 - 1, dtype=np.float32)
    if rawmode == 'L':
        return np.divide(pil_png, 2 ** 8 - 1, dtype=np.float32)
    if rawmode == 'I;16B':
        return np.divide(pil_png, 2 ** 16 - 1, dtype=np.float32)
    if mode == 'RGB':
        return np.divide(pil_png, 2 ** 8 - 1, dtype=np.float32)
    if mode == 'P':
        return np.divide(pil_png.convert('RGBA'), 2 ** 8 - 1, dtype=np.float32)
    if mode == 'LA':
        return np.divide(pil_png.convert('RGBA'), 2 ** 8 - 1, dtype=np.float32)
    if mode == 'RGBA':
        return np.divide(pil_png, 2 ** 8 - 1, dtype=np.float32)
    raise ValueError(f'Unknown PIL rawmode: {rawmode}')