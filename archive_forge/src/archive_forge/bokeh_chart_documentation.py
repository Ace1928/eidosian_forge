from __future__ import annotations
import hashlib
import json
from typing import TYPE_CHECKING, Final, cast
from streamlit.errors import StreamlitAPIException
from streamlit.proto.BokehChart_pb2 import BokehChart as BokehChartProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.util import HASHLIB_KWARGS
Get our DeltaGenerator.