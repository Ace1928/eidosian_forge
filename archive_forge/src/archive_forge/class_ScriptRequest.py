from __future__ import annotations
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import cast
from streamlit import util
from streamlit.proto.WidgetStates_pb2 import WidgetStates
from streamlit.runtime.state import coalesce_widget_states
@dataclass(frozen=True)
class ScriptRequest:
    """A STOP or RERUN request and associated data."""
    type: ScriptRequestType
    _rerun_data: RerunData | None = None

    @property
    def rerun_data(self) -> RerunData:
        if self.type is not ScriptRequestType.RERUN:
            raise RuntimeError('RerunData is only set for RERUN requests.')
        return cast(RerunData, self._rerun_data)

    def __repr__(self) -> str:
        return util.repr_(self)