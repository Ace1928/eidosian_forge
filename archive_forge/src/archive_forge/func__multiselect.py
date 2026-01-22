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
def _multiselect(self, label: str, options: OptionSequence[T], default: Sequence[Any] | Any | None=None, format_func: Callable[[Any], Any]=str, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, max_selections: int | None=None, placeholder: str='Choose an option', disabled: bool=False, label_visibility: LabelVisibility='visible', ctx: ScriptRunContext | None=None) -> list[T]:
    key = to_key(key)
    check_callback_rules(self.dg, on_change)
    check_session_state_rules(default_value=default, key=key)
    opt = ensure_indexable(options)
    check_python_comparable(opt)
    maybe_raise_label_warnings(label, label_visibility)
    indices = _check_and_convert_to_indices(opt, default)
    id = compute_widget_id('multiselect', user_key=key, label=label, options=[str(format_func(option)) for option in opt], default=indices, key=key, help=help, max_selections=max_selections, placeholder=placeholder, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
    default_value: list[int] = [] if indices is None else indices
    multiselect_proto = MultiSelectProto()
    multiselect_proto.id = id
    multiselect_proto.label = label
    multiselect_proto.default[:] = default_value
    multiselect_proto.options[:] = [str(format_func(option)) for option in opt]
    multiselect_proto.form_id = current_form_id(self.dg)
    multiselect_proto.max_selections = max_selections or 0
    multiselect_proto.placeholder = placeholder
    multiselect_proto.disabled = disabled
    multiselect_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
    if help is not None:
        multiselect_proto.help = dedent(help)
    serde = MultiSelectSerde(opt, default_value)
    widget_state = register_widget('multiselect', multiselect_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
    default_count = _get_default_count(widget_state.value)
    if max_selections and default_count > max_selections:
        raise StreamlitAPIException(_get_over_max_options_message(default_count, max_selections))
    widget_state = maybe_coerce_enum_sequence(widget_state, options, opt)
    if widget_state.value_changed:
        multiselect_proto.value[:] = serde.serialize(widget_state.value)
        multiselect_proto.set_value = True
    if ctx:
        save_for_app_testing(ctx, id, format_func)
    self.dg._enqueue('multiselect', multiselect_proto)
    return widget_state.value