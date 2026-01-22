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
def _text_area(self, label: str, value: SupportsStr | None='', height: int | None=None, max_chars: int | None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible', ctx: ScriptRunContext | None=None) -> str | None:
    key = to_key(key)
    check_callback_rules(self.dg, on_change)
    check_session_state_rules(default_value=None if value == '' else value, key=key)
    maybe_raise_label_warnings(label, label_visibility)
    value = str(value) if value is not None else None
    id = compute_widget_id('text_area', user_key=key, label=label, value=value, height=height, max_chars=max_chars, key=key, help=help, placeholder=str(placeholder), form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
    text_area_proto = TextAreaProto()
    text_area_proto.id = id
    text_area_proto.label = label
    if value is not None:
        text_area_proto.default = value
    text_area_proto.form_id = current_form_id(self.dg)
    text_area_proto.disabled = disabled
    text_area_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
    if help is not None:
        text_area_proto.help = dedent(help)
    if height is not None:
        text_area_proto.height = height
    if max_chars is not None:
        text_area_proto.max_chars = max_chars
    if placeholder is not None:
        text_area_proto.placeholder = str(placeholder)
    serde = TextAreaSerde(value)
    widget_state = register_widget('text_area', text_area_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
    if widget_state.value_changed:
        if widget_state.value is not None:
            text_area_proto.value = widget_state.value
        text_area_proto.set_value = True
    self.dg._enqueue('text_area', text_area_proto)
    return widget_state.value