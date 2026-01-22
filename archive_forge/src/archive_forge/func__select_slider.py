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
def _select_slider(self, label: str, options: OptionSequence[T]=(), value: object=None, format_func: Callable[[Any], Any]=str, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible', ctx: ScriptRunContext | None=None) -> T | tuple[T, T]:
    key = to_key(key)
    check_callback_rules(self.dg, on_change)
    check_session_state_rules(default_value=value, key=key)
    maybe_raise_label_warnings(label, label_visibility)
    opt = ensure_indexable(options)
    check_python_comparable(opt)
    if len(opt) == 0:
        raise StreamlitAPIException('The `options` argument needs to be non-empty')

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
    slider_value = as_index_list(value)
    id = compute_widget_id('select_slider', user_key=key, label=label, options=[str(format_func(option)) for option in opt], value=slider_value, key=key, help=help, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
    slider_proto = SliderProto()
    slider_proto.id = id
    slider_proto.type = SliderProto.Type.SELECT_SLIDER
    slider_proto.label = label
    slider_proto.format = '%s'
    slider_proto.default[:] = slider_value
    slider_proto.min = 0
    slider_proto.max = len(opt) - 1
    slider_proto.step = 1
    slider_proto.data_type = SliderProto.INT
    slider_proto.options[:] = [str(format_func(option)) for option in opt]
    slider_proto.form_id = current_form_id(self.dg)
    slider_proto.disabled = disabled
    slider_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
    if help is not None:
        slider_proto.help = dedent(help)
    serde = SelectSliderSerde(opt, slider_value, _is_range_value(value))
    widget_state = register_widget('slider', slider_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
    if isinstance(widget_state.value, tuple):
        widget_state = maybe_coerce_enum_sequence(cast(RegisterWidgetResult[Tuple[T, T]], widget_state), options, opt)
    else:
        widget_state = maybe_coerce_enum(widget_state, options, opt)
    if widget_state.value_changed:
        slider_proto.value[:] = serde.serialize(widget_state.value)
        slider_proto.set_value = True
    if ctx:
        save_for_app_testing(ctx, id, format_func)
    self.dg._enqueue('slider', slider_proto)
    return widget_state.value