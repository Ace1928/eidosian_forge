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
def _check_serializable(self) -> None:
    """Verify that everything added to session state can be serialized.
        We use pickleability as the metric for serializability, and test for
        pickleability by just trying it.
        """
    for k in self:
        try:
            pickle.dumps(self[k])
        except Exception as e:
            err_msg = f"Cannot serialize the value (of type `{type(self[k])}`) of '{k}' in st.session_state.\n                Streamlit has been configured to use [pickle](https://docs.python.org/3/library/pickle.html) to\n                serialize session_state values. Please convert the value to a pickle-serializable type. To learn\n                more about this behavior, see [our docs](https://docs.streamlit.io/knowledge-base/using-streamlit/serializable-session-state). "
            raise UnserializableSessionStateError(err_msg) from e