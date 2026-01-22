from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Generic, Sequence, cast
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Selectbox_pb2 import Selectbox as SelectboxProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id, save_for_app_testing
from streamlit.type_util import (
from streamlit.util import index_
def _selectbox(self, label: str, options: OptionSequence[T], index: int | None=0, format_func: Callable[[Any], Any]=str, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str='Choose an option', disabled: bool=False, label_visibility: LabelVisibility='visible', ctx: ScriptRunContext | None=None) -> T | None:
    key = to_key(key)
    check_callback_rules(self.dg, on_change)
    check_session_state_rules(default_value=None if index == 0 else index, key=key)
    maybe_raise_label_warnings(label, label_visibility)
    opt = ensure_indexable(options)
    check_python_comparable(opt)
    id = compute_widget_id('selectbox', user_key=key, label=label, options=[str(format_func(option)) for option in opt], index=index, key=key, help=help, placeholder=placeholder, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
    if not isinstance(index, int) and index is not None:
        raise StreamlitAPIException('Selectbox Value has invalid type: %s' % type(index).__name__)
    if index is not None and len(opt) > 0 and (not 0 <= index < len(opt)):
        raise StreamlitAPIException('Selectbox index must be between 0 and length of options')
    selectbox_proto = SelectboxProto()
    selectbox_proto.id = id
    selectbox_proto.label = label
    if index is not None:
        selectbox_proto.default = index
    selectbox_proto.options[:] = [str(format_func(option)) for option in opt]
    selectbox_proto.form_id = current_form_id(self.dg)
    selectbox_proto.placeholder = placeholder
    selectbox_proto.disabled = disabled
    selectbox_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
    if help is not None:
        selectbox_proto.help = dedent(help)
    serde = SelectboxSerde(opt, index)
    widget_state = register_widget('selectbox', selectbox_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
    widget_state = maybe_coerce_enum(widget_state, options, opt)
    if widget_state.value_changed:
        serialized_value = serde.serialize(widget_state.value)
        if serialized_value is not None:
            selectbox_proto.value = serialized_value
        selectbox_proto.set_value = True
    if ctx:
        save_for_app_testing(ctx, id, format_func)
    self.dg._enqueue('selectbox', selectbox_proto)
    return widget_state.value