import base64
from collections.abc import Sized, Sequence, Mapping
import functools
import importlib
import inspect
import io
import itertools
from numbers import Real
import re
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale
from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS
def _repr_png_(self):
    """Generate a PNG representation of the Colormap."""
    X = np.tile(np.linspace(0, 1, _REPR_PNG_SIZE[0]), (_REPR_PNG_SIZE[1], 1))
    pixels = self(X, bytes=True)
    png_bytes = io.BytesIO()
    title = self.name + ' colormap'
    author = f'Matplotlib v{mpl.__version__}, https://matplotlib.org'
    pnginfo = PngInfo()
    pnginfo.add_text('Title', title)
    pnginfo.add_text('Description', title)
    pnginfo.add_text('Author', author)
    pnginfo.add_text('Software', author)
    Image.fromarray(pixels).save(png_bytes, format='png', pnginfo=pnginfo)
    return png_bytes.getvalue()