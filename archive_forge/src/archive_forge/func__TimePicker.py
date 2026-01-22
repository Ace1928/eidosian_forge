import typing
from typing import Any, Dict, Union
import ipyvue
import ipywidgets
import reacton
from reacton import ipywidgets as w
from reacton.core import Element, ValueElement
from reacton.utils import implements
import ipyvuetify
def _TimePicker(allowed_hours: list=None, allowed_minutes: list=None, allowed_seconds: list=None, ampm_in_title: bool=None, attributes: dict={}, children: list=[], class_: str=None, color: str=None, dark: bool=None, disabled: bool=None, format: str=None, full_width: bool=None, header_color: str=None, landscape: bool=None, layout: Union[Dict[str, Any], Element[ipywidgets.widgets.widget_layout.Layout]]={}, light: bool=None, max: str=None, min: str=None, no_title: bool=None, readonly: bool=None, scrollable: bool=None, slot: str=None, style_: str=None, tabbable: bool=None, tooltip: str=None, use_seconds: bool=None, v_model: Any='!!disabled!!', v_on: str=None, v_slots: list=[], value: Any=None, width: typing.Union[float, str]=None, on_v_model: typing.Callable[[Any], Any]=None) -> ValueElement[ipyvuetify.generated.TimePicker, Any]:
    """
    :param tabbable: Is widget tabbable?
    :param tooltip: A tooltip caption.
    """
    ...