from __future__ import annotations
import numbers
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Literal, Union, cast, overload
from typing_extensions import TypeAlias
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.js_number import JSNumber, JSNumberBoundsException
from streamlit.proto.NumberInput_pb2 import NumberInput as NumberInputProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
@dataclass
class NumberInputSerde:
    value: Number | None
    data_type: int

    def serialize(self, v: Number | None) -> Number | None:
        return v

    def deserialize(self, ui_value: Number | None, widget_id: str='') -> Number | None:
        val: Number | None = ui_value if ui_value is not None else self.value
        if val is not None and self.data_type == NumberInputProto.INT:
            val = int(val)
        return val