from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from numbers import Integral, Real
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Final, Sequence, Tuple, TypeVar, Union, cast
from typing_extensions import TypeAlias
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.js_number import JSNumber, JSNumberBoundsException
from streamlit.proto.Slider_pb2 import Slider as SliderProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
def _slider(self, label: str, min_value=None, max_value=None, value=None, step: Step | None=None, format: str | None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible', ctx: ScriptRunContext | None=None) -> SliderReturn:
    key = to_key(key)
    check_callback_rules(self.dg, on_change)
    check_session_state_rules(default_value=value, key=key)
    maybe_raise_label_warnings(label, label_visibility)
    id = compute_widget_id('slider', user_key=key, label=label, min_value=min_value, max_value=max_value, value=value, step=step, format=format, key=key, help=help, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
    SUPPORTED_TYPES = {Integral: SliderProto.INT, Real: SliderProto.FLOAT, datetime: SliderProto.DATETIME, date: SliderProto.DATE, time: SliderProto.TIME}
    TIMELIKE_TYPES = (SliderProto.DATETIME, SliderProto.TIME, SliderProto.DATE)
    if value is None:
        single_value = True
        session_state = get_session_state().filtered_state
        if key is not None and key in session_state:
            state_value = session_state[key]
            single_value = isinstance(state_value, tuple(SUPPORTED_TYPES.keys()))
        if single_value:
            value = min_value if min_value is not None else 0
        else:
            mn = min_value if min_value is not None else 0
            mx = max_value if max_value is not None else 100
            value = [mn, mx]
    single_value = isinstance(value, tuple(SUPPORTED_TYPES.keys()))
    range_value = isinstance(value, (list, tuple)) and len(value) in (0, 1, 2)
    if not single_value and (not range_value):
        raise StreamlitAPIException('Slider value should either be an int/float/datetime or a list/tuple of 0 to 2 ints/floats/datetimes')
    if single_value:
        value = [value]

    def value_to_generic_type(v):
        if isinstance(v, Integral):
            return SUPPORTED_TYPES[Integral]
        elif isinstance(v, Real):
            return SUPPORTED_TYPES[Real]
        else:
            return SUPPORTED_TYPES[type(v)]

    def all_same_type(items):
        return len(set(map(value_to_generic_type, items))) < 2
    if not all_same_type(value):
        raise StreamlitAPIException(f'Slider tuple/list components must be of the same type.\nBut were: {list(map(type, value))}')
    if len(value) == 0:
        data_type = SliderProto.INT
    else:
        data_type = value_to_generic_type(value[0])
    datetime_min = time.min
    datetime_max = time.max
    if data_type == SliderProto.TIME:
        datetime_min = time.min.replace(tzinfo=value[0].tzinfo)
        datetime_max = time.max.replace(tzinfo=value[0].tzinfo)
    if data_type in (SliderProto.DATETIME, SliderProto.DATE):
        datetime_min = value[0] - timedelta(days=14)
        datetime_max = value[0] + timedelta(days=14)
    DEFAULTS = {SliderProto.INT: {'min_value': 0, 'max_value': 100, 'step': 1, 'format': '%d'}, SliderProto.FLOAT: {'min_value': 0.0, 'max_value': 1.0, 'step': 0.01, 'format': '%0.2f'}, SliderProto.DATETIME: {'min_value': datetime_min, 'max_value': datetime_max, 'step': timedelta(days=1), 'format': 'YYYY-MM-DD'}, SliderProto.DATE: {'min_value': datetime_min, 'max_value': datetime_max, 'step': timedelta(days=1), 'format': 'YYYY-MM-DD'}, SliderProto.TIME: {'min_value': datetime_min, 'max_value': datetime_max, 'step': timedelta(minutes=15), 'format': 'HH:mm'}}
    if min_value is None:
        min_value = DEFAULTS[data_type]['min_value']
    if max_value is None:
        max_value = DEFAULTS[data_type]['max_value']
    if step is None:
        step = cast(Step, DEFAULTS[data_type]['step'])
        if data_type in (SliderProto.DATETIME, SliderProto.DATE) and max_value - min_value < timedelta(days=1):
            step = timedelta(minutes=15)
    if format is None:
        format = cast(str, DEFAULTS[data_type]['format'])
    if step == 0:
        raise StreamlitAPIException('Slider components cannot be passed a `step` of 0.')
    slider_args = [min_value, max_value, step]
    int_args = all(map(lambda a: isinstance(a, Integral), slider_args))
    float_args = all(map(lambda a: isinstance(a, Real) and (not isinstance(a, Integral)), slider_args))
    timelike_args = data_type in TIMELIKE_TYPES and isinstance(step, timedelta) and (type(min_value) == type(max_value))
    if not int_args and (not float_args) and (not timelike_args):
        raise StreamlitAPIException('Slider value arguments must be of matching types.\n`min_value` has %(min_type)s type.\n`max_value` has %(max_type)s type.\n`step` has %(step)s type.' % {'min_type': type(min_value).__name__, 'max_type': type(max_value).__name__, 'step': type(step).__name__})
    all_ints = data_type == SliderProto.INT and int_args
    all_floats = data_type == SliderProto.FLOAT and float_args
    all_timelikes = data_type in TIMELIKE_TYPES and timelike_args
    if not all_ints and (not all_floats) and (not all_timelikes):
        raise StreamlitAPIException('Both value and arguments must be of the same type.\n`value` has %(value_type)s type.\n`min_value` has %(min_type)s type.\n`max_value` has %(max_type)s type.' % {'value_type': type(value).__name__, 'min_type': type(min_value).__name__, 'max_type': type(max_value).__name__})
    min_value = min(min_value, max_value)
    max_value = max(min_value, max_value)
    if len(value) == 1:
        min_value = min(value[0], min_value)
        max_value = max(value[0], max_value)
    elif len(value) == 2:
        start, end = value
        if start > end:
            start, end = (end, start)
            value = (start, end)
        min_value = min(start, min_value)
        max_value = max(end, max_value)
    else:
        value = [min_value, max_value]
    try:
        if all_ints:
            JSNumber.validate_int_bounds(min_value, '`min_value`')
            JSNumber.validate_int_bounds(max_value, '`max_value`')
        elif all_floats:
            JSNumber.validate_float_bounds(min_value, '`min_value`')
            JSNumber.validate_float_bounds(max_value, '`max_value`')
        elif all_timelikes:
            pass
    except JSNumberBoundsException as e:
        raise StreamlitAPIException(str(e))
    orig_tz = None
    if data_type == SliderProto.TIME:
        value = list(map(_time_to_datetime, value))
        min_value = _time_to_datetime(min_value)
        max_value = _time_to_datetime(max_value)
    if data_type == SliderProto.DATE:
        value = list(map(_date_to_datetime, value))
        min_value = _date_to_datetime(min_value)
        max_value = _date_to_datetime(max_value)
    if data_type in TIMELIKE_TYPES:
        orig_tz = value[0].tzinfo if data_type in (SliderProto.TIME, SliderProto.DATETIME) else None
        value = list(map(_datetime_to_micros, value))
        min_value = _datetime_to_micros(min_value)
        max_value = _datetime_to_micros(max_value)
        step = _delta_to_micros(cast(timedelta, step))
    slider_proto = SliderProto()
    slider_proto.type = SliderProto.Type.SLIDER
    slider_proto.id = id
    slider_proto.label = label
    slider_proto.format = format
    slider_proto.default[:] = value
    slider_proto.min = min_value
    slider_proto.max = max_value
    slider_proto.step = cast(float, step)
    slider_proto.data_type = data_type
    slider_proto.options[:] = []
    slider_proto.form_id = current_form_id(self.dg)
    slider_proto.disabled = disabled
    slider_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
    if help is not None:
        slider_proto.help = dedent(help)
    serde = SliderSerde(value, data_type, single_value, orig_tz)
    widget_state = register_widget('slider', slider_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
    if widget_state.value_changed:
        slider_proto.value[:] = serde.serialize(widget_state.value)
        slider_proto.set_value = True
    self.dg._enqueue('slider', slider_proto)
    return cast(SliderReturn, widget_state.value)