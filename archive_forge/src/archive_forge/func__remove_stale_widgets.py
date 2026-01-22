from __future__ import annotations
import json
import pickle
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import (
from typing_extensions import TypeAlias
import streamlit as st
from streamlit import config, util
from streamlit.errors import StreamlitAPIException, UnserializableSessionStateError
from streamlit.proto.WidgetStates_pb2 import WidgetState as WidgetStateProto
from streamlit.proto.WidgetStates_pb2 import WidgetStates as WidgetStatesProto
from streamlit.runtime.state.common import (
from streamlit.runtime.state.query_params import QueryParams
from streamlit.runtime.stats import CacheStat, CacheStatsProvider, group_stats
from streamlit.type_util import ValueFieldName, is_array_value_field_name
def _remove_stale_widgets(self, active_widget_ids: set[str]) -> None:
    """Remove widget state for widgets whose ids aren't in `active_widget_ids`."""
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    ctx = get_script_run_ctx()
    if ctx is None:
        return
    self._new_widget_state.remove_stale_widgets(active_widget_ids, ctx.fragment_ids_this_run)
    self._old_state = {k: v for k, v in self._old_state.items() if not is_widget_id(k) or not _is_stale_widget(self._new_widget_state.widget_metadata.get(k), active_widget_ids, ctx.fragment_ids_this_run)}