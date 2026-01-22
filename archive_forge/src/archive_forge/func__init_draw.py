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
def _init_draw(self):
    super()._init_draw()
    if self._init_func is None:
        try:
            frame_data = next(self.new_frame_seq())
        except StopIteration:
            warnings.warn('Can not start iterating the frames for the initial draw. This can be caused by passing in a 0 length sequence for *frames*.\n\nIf you passed *frames* as a generator it may be exhausted due to a previous display or save.')
            return
        self._draw_frame(frame_data)
    else:
        self._drawn_artists = self._init_func()
        if self._blit:
            if self._drawn_artists is None:
                raise RuntimeError('The init_func must return a sequence of Artist objects.')
            for a in self._drawn_artists:
                a.set_animated(self._blit)
    self._save_seq = []