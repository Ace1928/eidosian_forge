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
def _check_and_convert_to_indices(opt: Sequence[Any], default_values: Sequence[Any] | Any | None) -> list[int] | None:
    """Perform validation checks and return indices based on the default values."""
    if default_values is None and None not in opt:
        return None
    if not isinstance(default_values, list):
        if is_type(default_values, 'numpy.ndarray') or is_type(default_values, 'pandas.core.series.Series'):
            default_values = list(cast(Sequence[Any], default_values))
        elif not default_values or default_values in opt:
            default_values = [default_values]
        else:
            default_values = list(default_values)
    for value in default_values:
        if value not in opt:
            raise StreamlitAPIException('Every Multiselect default value must exist in options')
    return [opt.index(value) for value in default_values]