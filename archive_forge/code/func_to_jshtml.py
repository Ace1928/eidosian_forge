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
def to_jshtml(self, fps=None, embed_frames=True, default_mode=None):
    """
        Generate HTML representation of the animation.

        Parameters
        ----------
        fps : int, optional
            Movie frame rate (per second). If not set, the frame rate from
            the animation's frame interval.
        embed_frames : bool, optional
        default_mode : str, optional
            What to do when the animation ends. Must be one of ``{'loop',
            'once', 'reflect'}``. Defaults to ``'loop'`` if the *repeat*
            parameter is True, otherwise ``'once'``.
        """
    if fps is None and hasattr(self, '_interval'):
        fps = 1000 / self._interval
    if default_mode is None:
        default_mode = 'loop' if getattr(self, '_repeat', False) else 'once'
    if not hasattr(self, '_html_representation'):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir, 'temp.html')
            writer = HTMLWriter(fps=fps, embed_frames=embed_frames, default_mode=default_mode)
            self.save(str(path), writer=writer)
            self._html_representation = path.read_text()
    return self._html_representation