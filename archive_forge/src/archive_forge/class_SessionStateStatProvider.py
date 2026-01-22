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
@dataclass
class SessionStateStatProvider(CacheStatsProvider):
    _session_mgr: SessionManager

    def get_stats(self) -> list[CacheStat]:
        stats: list[CacheStat] = []
        for session_info in self._session_mgr.list_active_sessions():
            session_state = session_info.session.session_state
            stats.extend(session_state.get_stats())
        return group_stats(stats)