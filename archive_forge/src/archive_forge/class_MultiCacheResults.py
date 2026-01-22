from __future__ import annotations
import contextlib
import hashlib
import threading
import types
from dataclasses import dataclass
from typing import (
from google.protobuf.message import Message
import streamlit as st
from streamlit import runtime, util
from streamlit.elements import NONWIDGET_ELEMENTS, WIDGETS
from streamlit.logger import get_logger
from streamlit.proto.Block_pb2 import Block
from streamlit.runtime.caching.cache_errors import (
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.hashing import update_hash
from streamlit.runtime.scriptrunner.script_run_context import (
from streamlit.runtime.state.common import WidgetMetadata
from streamlit.util import HASHLIB_KWARGS
@dataclass
class MultiCacheResults:
    """Widgets called by a cache-decorated function, and a mapping of the
    widget-derived cache key to the final results of executing the function.
    """
    widget_ids: set[str]
    results: dict[str, CachedResult]

    def get_current_widget_key(self, ctx: ScriptRunContext, cache_type: CacheType) -> str:
        state = ctx.session_state
        widget_values = [(wid, state[wid]) for wid in sorted(self.widget_ids) if wid in state]
        widget_key = _make_widget_key(widget_values, cache_type)
        return widget_key