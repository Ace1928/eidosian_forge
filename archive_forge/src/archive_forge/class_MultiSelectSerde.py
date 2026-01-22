from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Generic, Sequence, cast, overload
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.MultiSelect_pb2 import MultiSelect as MultiSelectProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id, save_for_app_testing
from streamlit.type_util import (
@dataclass
class MultiSelectSerde(Generic[T]):
    options: Sequence[T]
    default_value: list[int]

    def serialize(self, value: list[T]) -> list[int]:
        return _check_and_convert_to_indices(self.options, value)

    def deserialize(self, ui_value: list[int] | None, widget_id: str='') -> list[T]:
        current_value: list[int] = ui_value if ui_value is not None else self.default_value
        return [self.options[i] for i in current_value]