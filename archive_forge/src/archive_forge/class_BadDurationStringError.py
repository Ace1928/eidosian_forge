from __future__ import annotations
import math
from datetime import timedelta
from typing import Any, Literal, overload
from streamlit import config
from streamlit.errors import MarkdownFormattedException, StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.forward_msg_cache import populate_hash_if_needed
class BadDurationStringError(StreamlitAPIException):
    """Raised when a bad duration argument string is passed."""

    def __init__(self, duration: str):
        MarkdownFormattedException.__init__(self, f"TTL string doesn't look right. It should be formatted as`'1d2h34m'` or `2 days`, for example. Got: {duration}")