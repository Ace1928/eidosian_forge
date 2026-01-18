from fractions import Fraction
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union, Generator
import av
import av.filter
import numpy as np
from numpy.lib.stride_tricks import as_strided
from ..core import Request
from ..core.request import URI_BYTES, InitializationError, IOMode
from ..core.v3_plugin_api import ImageProperties, PluginV3
@property
def video_stream_metadata(self):
    """Stream-specific metadata.

        A dictionary containing metadata stored at the stream level.

        """
    return self._video_stream.metadata