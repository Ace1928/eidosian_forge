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
class ImageMagickBase:
    """
    Mixin class for ImageMagick output.

    This is a base class for the concrete `ImageMagickWriter` and
    `ImageMagickFileWriter` classes, which define an ``input_names`` attribute
    (or property) specifying the input names passed to ImageMagick.
    """
    _exec_key = 'animation.convert_path'
    _args_key = 'animation.convert_args'

    def _args(self):
        fmt = 'rgba' if self.frame_format == 'raw' else self.frame_format
        extra_args = self.extra_args if self.extra_args is not None else mpl.rcParams[self._args_key]
        return [self.bin_path(), '-size', '%ix%i' % self.frame_size, '-depth', '8', '-delay', str(100 / self.fps), '-loop', '0', f'{fmt}:{self.input_names}', *extra_args, self.outfile]

    @classmethod
    def bin_path(cls):
        binpath = super().bin_path()
        if binpath == 'convert':
            binpath = mpl._get_executable_info('magick').executable
        return binpath

    @classmethod
    def isAvailable(cls):
        try:
            return super().isAvailable()
        except mpl.ExecutableNotFoundError as _enf:
            _log.debug('ImageMagick unavailable due to: %s', _enf)
            return False