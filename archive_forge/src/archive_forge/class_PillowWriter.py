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
@writers.register('pillow')
class PillowWriter(AbstractMovieWriter):

    @classmethod
    def isAvailable(cls):
        return True

    def setup(self, fig, outfile, dpi=None):
        super().setup(fig, outfile, dpi=dpi)
        self._frames = []

    def grab_frame(self, **savefig_kwargs):
        _validate_grabframe_kwargs(savefig_kwargs)
        buf = BytesIO()
        self.fig.savefig(buf, **{**savefig_kwargs, 'format': 'rgba', 'dpi': self.dpi})
        self._frames.append(Image.frombuffer('RGBA', self.frame_size, buf.getbuffer(), 'raw', 'RGBA', 0, 1))

    def finish(self):
        self._frames[0].save(self.outfile, save_all=True, append_images=self._frames[1:], duration=int(1000 / self.fps), loop=0)