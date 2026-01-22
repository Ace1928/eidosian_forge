import functools
import logging
import math
import pathlib
import sys
import weakref
import numpy as np
import PIL.Image
import matplotlib as mpl
from matplotlib.backend_bases import (
from matplotlib import _api, cbook, backend_tools
from matplotlib._pylab_helpers import Gcf
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
import wx
def _get_imagesave_wildcards(self):
    """Return the wildcard string for the filesave dialog."""
    default_filetype = self.get_default_filetype()
    filetypes = self.get_supported_filetypes_grouped()
    sorted_filetypes = sorted(filetypes.items())
    wildcards = []
    extensions = []
    filter_index = 0
    for i, (name, exts) in enumerate(sorted_filetypes):
        ext_list = ';'.join(['*.%s' % ext for ext in exts])
        extensions.append(exts[0])
        wildcard = f'{name} ({ext_list})|{ext_list}'
        if default_filetype in exts:
            filter_index = i
        wildcards.append(wildcard)
    wildcards = '|'.join(wildcards)
    return (wildcards, extensions, filter_index)