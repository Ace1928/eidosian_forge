from __future__ import annotations
from collections.abc import Iterable, Mapping
from inspect import Parameter
from numbers import Integral, Number, Real
from typing import Any, Optional, Tuple
import param
from .base import Widget
from .input import Checkbox, TextInput
from .select import Select
from .slider import DiscreteSlider, FloatSlider, IntSlider
@staticmethod
def widget_from_iterable(o, name):
    """Make widgets from an iterable. This should not be done for
        a string or tuple."""
    values = list(o.values()) if isinstance(o, Mapping) else list(o)
    widget_type = DiscreteSlider if all((param._is_number(v) for v in values)) else Select
    if isinstance(o, (list, dict)):
        return widget_type(options=o, name=name)
    elif isinstance(o, Mapping):
        return widget_type(options=list(o.items()), name=name)
    else:
        return widget_type(options=list(o), name=name)