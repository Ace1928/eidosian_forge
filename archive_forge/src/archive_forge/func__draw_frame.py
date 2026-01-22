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
def _draw_frame(self, framedata):
    if self._cache_frame_data:
        self._save_seq.append(framedata)
        self._save_seq = self._save_seq[-self._save_count:]
    self._drawn_artists = self._func(framedata, *self._args)
    if self._blit:
        err = RuntimeError('The animation function must return a sequence of Artist objects.')
        try:
            iter(self._drawn_artists)
        except TypeError:
            raise err from None
        for i in self._drawn_artists:
            if not isinstance(i, mpl.artist.Artist):
                raise err
        self._drawn_artists = sorted(self._drawn_artists, key=lambda x: x.get_zorder())
        for a in self._drawn_artists:
            a.set_animated(self._blit)