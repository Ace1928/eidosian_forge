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
Standardized ndimage metadata
        Parameters
        ----------
        index : int
            If the ImageResource contains multiple ndimages, and index is an
            integer, select the index-th ndimage from among them and return its
            properties. If index is an ellipsis (...), read and return the
            properties of all ndimages in the file stacked along a new batch
            dimension. If index is None, this plugin reads and returns the
            properties of the first image (index=0) unless the image is a GIF or
            APNG, in which case it reads and returns the properties all images
            (index=...).

        Returns
        -------
        properties : ImageProperties
            A dataclass filled with standardized image metadata.

        Notes
        -----
        This does not decode pixel data and is fast for large images.

        