from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from textwrap import dedent
from typing import (
from typing_extensions import TypeAlias
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.DateInput_pb2 import DateInput as DateInputProto
from streamlit.proto.TimeInput_pb2 import TimeInput as TimeInputProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.time_util import adjust_years
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
@dataclass
class DateInputSerde:
    value: _DateInputValues

    def deserialize(self, ui_value: Any, widget_id: str='') -> DateWidgetReturn:
        return_value: Sequence[date] | None
        if ui_value is not None:
            return_value = tuple((datetime.strptime(v, '%Y/%m/%d').date() for v in ui_value))
        else:
            return_value = self.value.value
        if return_value is None or len(return_value) == 0:
            return () if self.value.is_range else None
        if not self.value.is_range:
            return return_value[0]
        return cast(DateWidgetReturn, tuple(return_value))

    def serialize(self, v: DateWidgetReturn) -> list[str]:
        if v is None:
            return []
        to_serialize = list(v) if isinstance(v, (list, tuple)) else [v]
        return [date.strftime(v, '%Y/%m/%d') for v in to_serialize]