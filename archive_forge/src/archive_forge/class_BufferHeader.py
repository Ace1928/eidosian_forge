from __future__ import annotations
import logging # isort:skip
import json
from typing import (
import bokeh.util.serialization as bkserial
from ..core.json_encoder import serialize_json
from ..core.serialization import Buffer, Serialized
from ..core.types import ID
from .exceptions import MessageError, ProtocolError
class BufferHeader(TypedDict):
    id: ID