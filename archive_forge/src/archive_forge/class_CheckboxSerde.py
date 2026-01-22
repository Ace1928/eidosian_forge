from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, cast
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.proto.Checkbox_pb2 import Checkbox as CheckboxProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
@dataclass
class CheckboxSerde:
    value: bool

    def serialize(self, v: bool) -> bool:
        return bool(v)

    def deserialize(self, ui_value: bool | None, widget_id: str='') -> bool:
        return bool(ui_value if ui_value is not None else self.value)