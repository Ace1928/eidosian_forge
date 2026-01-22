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
def _clip_image(image: npt.NDArray[Any], clamp: bool) -> npt.NDArray[Any]:
    import numpy as np
    data = image
    if issubclass(image.dtype.type, np.floating):
        if clamp:
            data = np.clip(image, 0, 1.0)
        elif np.amin(image) < 0.0 or np.amax(image) > 1.0:
            raise RuntimeError('Data is outside [0.0, 1.0] and clamp is not set.')
        data = data * 255
    elif clamp:
        data = np.clip(image, 0, 255)
    elif np.amin(image) < 0 or np.amax(image) > 255:
        raise RuntimeError('Data is outside [0, 255] and clamp is not set.')
    return data