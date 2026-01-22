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
def _is_stale_widget(metadata: WidgetMetadata[Any] | None, active_widget_ids: set[str], fragment_ids_this_run: set[str] | None) -> bool:
    if not metadata:
        return True
    elif metadata.id in active_widget_ids:
        return False
    elif fragment_ids_this_run and metadata.fragment_id not in fragment_ids_this_run:
        return False
    return True