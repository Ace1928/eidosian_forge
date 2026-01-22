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
def _ensure_image_size_and_format(image_data: bytes, width: int, image_format: ImageFormat) -> bytes:
    """Resize an image if it exceeds the given width, or if exceeds
    MAXIMUM_CONTENT_WIDTH. Ensure the image's format corresponds to the given
    ImageFormat. Return the (possibly resized and reformatted) image bytes.
    """
    from PIL import Image
    pil_image = Image.open(io.BytesIO(image_data))
    actual_width, actual_height = pil_image.size
    if width < 0 and actual_width > MAXIMUM_CONTENT_WIDTH:
        width = MAXIMUM_CONTENT_WIDTH
    if width > 0 and actual_width > width:
        new_height = int(1.0 * actual_height * width / actual_width)
        pil_image = pil_image.resize((width, new_height), resample=Image.Resampling.BILINEAR)
        return _PIL_to_bytes(pil_image, format=image_format, quality=90)
    if pil_image.format != image_format:
        return _PIL_to_bytes(pil_image, format=image_format, quality=90)
    return image_data