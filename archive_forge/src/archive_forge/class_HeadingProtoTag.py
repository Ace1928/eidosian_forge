from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Literal, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Heading_pb2 import Heading as HeadingProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.string_util import clean_text
from streamlit.type_util import SupportsStr
class HeadingProtoTag(Enum):
    TITLE_TAG = 'h1'
    HEADER_TAG = 'h2'
    SUBHEADER_TAG = 'h3'