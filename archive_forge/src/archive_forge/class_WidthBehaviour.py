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
class WidthBehaviour(IntEnum):
    """
    Special values that are recognized by the frontend and allow us to change the
    behavior of the displayed image.
    """
    ORIGINAL = -1
    COLUMN = -2
    AUTO = -3