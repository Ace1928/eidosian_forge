import sys
import warnings
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, cast
import numpy as np
from PIL import ExifTags, GifImagePlugin, Image, ImageSequence, UnidentifiedImageError
from PIL import __version__ as pil_version  # type: ignore
from ..core.request import URI_BYTES, InitializationError, IOMode, Request
from ..core.v3_plugin_api import ImageProperties, PluginV3
from ..typing import ArrayLike
def _exif_orientation_transform(orientation: int, mode: str) -> Callable:
    axis = -2 if Image.getmodebands(mode) > 1 else -1
    EXIF_ORIENTATION = {1: lambda x: x, 2: lambda x: np.flip(x, axis=axis), 3: lambda x: np.rot90(x, k=2), 4: lambda x: np.flip(x, axis=axis - 1), 5: lambda x: np.flip(np.rot90(x, k=3), axis=axis), 6: lambda x: np.rot90(x, k=3), 7: lambda x: np.flip(np.rot90(x, k=1), axis=axis), 8: lambda x: np.rot90(x, k=1)}
    return EXIF_ORIENTATION[orientation]