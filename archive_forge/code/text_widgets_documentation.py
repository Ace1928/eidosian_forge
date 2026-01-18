from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Literal, cast, overload
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.TextArea_pb2 import TextArea as TextAreaProto
from streamlit.proto.TextInput_pb2 import TextInput as TextInputProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import (
Get our DeltaGenerator.