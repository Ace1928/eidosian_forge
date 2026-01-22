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
@property
def _formats(self):
    available_formats = list()
    for config in known_plugins.values():
        with contextlib.suppress(ImportError):
            if config.is_legacy and config.format is not None:
                available_formats.append(config)
    return available_formats