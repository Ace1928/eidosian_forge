from __future__ import annotations
import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from html import escape
from importlib.resources import files
from xarray.core.formatting import (
from xarray.core.options import _get_boolean_with_default
def array_section(obj) -> str:
    data_id = 'section-' + str(uuid.uuid4())
    collapsed = 'checked' if _get_boolean_with_default('display_expand_data', default=True) else ''
    variable = getattr(obj, 'variable', obj)
    preview = escape(inline_variable_array_repr(variable, max_width=70))
    data_repr = short_data_repr_html(obj)
    data_icon = _icon('icon-database')
    return f"<div class='xr-array-wrap'><input id='{data_id}' class='xr-array-in' type='checkbox' {collapsed}><label for='{data_id}' title='Show/hide data repr'>{data_icon}</label><div class='xr-array-preview xr-preview'><span>{preview}</span></div><div class='xr-array-data'>{data_repr}</div></div>"