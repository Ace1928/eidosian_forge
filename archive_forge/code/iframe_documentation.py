from __future__ import annotations
from typing import TYPE_CHECKING, cast
from streamlit.proto.IFrame_pb2 import IFrame as IFrameProto
from streamlit.runtime.metrics_util import gather_metrics
Get our DeltaGenerator.