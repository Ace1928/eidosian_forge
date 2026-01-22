from __future__ import annotations
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import cast
from streamlit import util
from streamlit.proto.WidgetStates_pb2 import WidgetStates
from streamlit.runtime.state import coalesce_widget_states
class ScriptRequestType(Enum):
    CONTINUE = 'CONTINUE'
    STOP = 'STOP'
    RERUN = 'RERUN'