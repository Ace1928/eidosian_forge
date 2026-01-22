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
def _format_to_dtype(format: av.VideoFormat) -> np.dtype:
    """Convert a pyAV video format into a numpy dtype"""
    if len(format.components) == 0:
        raise ValueError(f"Can't determine dtype from format `{format.name}`. It has no channels.")
    endian = '>' if format.is_big_endian else '<'
    dtype = 'f' if 'f32' in format.name else 'u'
    bits_per_channel = [x.bits for x in format.components]
    n_bytes = str(int(ceil(bits_per_channel[0] / 8)))
    return np.dtype(endian + dtype + n_bytes)