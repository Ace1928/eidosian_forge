import abc
import base64
import contextlib
from io import BytesIO, TextIOWrapper
import itertools
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import uuid
import warnings
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib._animation_data import (
from matplotlib import _api, cbook
import matplotlib.colors as mcolors
def _adjust_frame_size(self):
    if self.codec == 'h264':
        wo, ho = self.fig.get_size_inches()
        w, h = adjusted_figsize(wo, ho, self.dpi, 2)
        if (wo, ho) != (w, h):
            self.fig.set_size_inches(w, h, forward=True)
            _log.info('figure size in inches has been adjusted from %s x %s to %s x %s', wo, ho, w, h)
    else:
        w, h = self.fig.get_size_inches()
    _log.debug('frame size in pixels is %s x %s', *self.frame_size)
    return (w, h)