import sys
import warnings
import contextlib
import numpy as np
from pathlib import Path
from . import Array, asarray
from .request import ImageMode
from ..config import known_plugins, known_extensions, PluginConfig, FileExtension
from ..config.plugins import _original_order
from .imopen import imopen
def _get_meta_data(self, index):
    """_get_meta_data(index)

            Plugins must implement this.

            It should return the meta data as a dict, corresponding to the
            given index, or to the file's (global) meta data if index is
            None.
            """
    raise NotImplementedError()