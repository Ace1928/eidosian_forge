from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Generic, Sequence, Tuple, cast
from typing_extensions import TypeGuard
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Slider_pb2 import Slider as SliderProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import (
from streamlit.type_util import (
from streamlit.util import index_
def as_index_list(v: object) -> list[int]:
    if _is_range_value(v):
        slider_value = [index_(opt, val) for val in v]
        start, end = slider_value
        if start > end:
            slider_value = [end, start]
        return slider_value
    else:
        try:
            return [index_(opt, v)]
        except ValueError:
            if value is not None:
                raise
            return [0]