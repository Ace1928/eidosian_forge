from __future__ import annotations
import io
import os
import re
from enum import IntEnum
from typing import TYPE_CHECKING, Final, List, Literal, Sequence, Union, cast
from typing_extensions import TypeAlias
from streamlit import runtime, url_util
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Image_pb2 import ImageList as ImageListProto
from streamlit.runtime import caching
from streamlit.runtime.metrics_util import gather_metrics
def _np_array_to_bytes(array: npt.NDArray[Any], output_format: str='JPEG') -> bytes:
    import numpy as np
    from PIL import Image
    img = Image.fromarray(array.astype(np.uint8))
    format = _validate_image_format_string(img, output_format)
    return _PIL_to_bytes(img, format)