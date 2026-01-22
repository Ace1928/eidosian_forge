from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Literal, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Heading_pb2 import Heading as HeadingProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.string_util import clean_text
from streamlit.type_util import SupportsStr
@staticmethod
def _create_heading_proto(tag: HeadingProtoTag, body: SupportsStr, anchor: Anchor=None, help: str | None=None, divider: Divider=False) -> HeadingProto:
    proto = HeadingProto()
    proto.tag = tag.value
    proto.body = clean_text(body)
    if divider:
        proto.divider = HeadingMixin._handle_divider_color(divider)
    if anchor is not None:
        if anchor is False:
            proto.hide_anchor = True
        elif isinstance(anchor, str):
            proto.anchor = anchor
        elif anchor is True:
            raise StreamlitAPIException('Anchor parameter has invalid value: %s. Supported values: None, any string or False' % anchor)
        else:
            raise StreamlitAPIException('Anchor parameter has invalid type: %s. Supported values: None, any string or False' % type(anchor).__name__)
    if help:
        proto.help = help
    return proto