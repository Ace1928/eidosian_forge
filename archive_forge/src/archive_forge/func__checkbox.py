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
def _checkbox(self, label: str, value: bool=False, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible', type: CheckboxProto.StyleType.ValueType=CheckboxProto.StyleType.DEFAULT, ctx: ScriptRunContext | None=None) -> bool:
    key = to_key(key)
    check_callback_rules(self.dg, on_change)
    check_session_state_rules(default_value=None if value is False else value, key=key)
    maybe_raise_label_warnings(label, label_visibility)
    id = compute_widget_id('toggle' if type == CheckboxProto.StyleType.TOGGLE else 'checkbox', user_key=key, label=label, value=bool(value), key=key, help=help, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
    checkbox_proto = CheckboxProto()
    checkbox_proto.id = id
    checkbox_proto.label = label
    checkbox_proto.default = bool(value)
    checkbox_proto.type = type
    checkbox_proto.form_id = current_form_id(self.dg)
    checkbox_proto.disabled = disabled
    checkbox_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
    if help is not None:
        checkbox_proto.help = dedent(help)
    serde = CheckboxSerde(value)
    checkbox_state = register_widget('checkbox', checkbox_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
    if checkbox_state.value_changed:
        checkbox_proto.value = checkbox_state.value
        checkbox_proto.set_value = True
    self.dg._enqueue('checkbox', checkbox_proto)
    return checkbox_state.value