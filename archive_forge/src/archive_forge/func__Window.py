import typing
from typing import Any, Dict, Union
import ipyvue
import ipywidgets
import reacton
from reacton import ipywidgets as w
from reacton.core import Element, ValueElement
from reacton.utils import implements
import ipyvuetify
def _Window(active_class: str=None, attributes: dict={}, children: list=[], class_: str=None, continuous: bool=None, dark: bool=None, layout: Union[Dict[str, Any], Element[ipywidgets.widgets.widget_layout.Layout]]={}, light: bool=None, mandatory: bool=None, max: typing.Union[float, str]=None, multiple: bool=None, next_icon: typing.Union[bool, str]=None, prev_icon: typing.Union[bool, str]=None, reverse: bool=None, show_arrows: bool=None, show_arrows_on_hover: bool=None, slot: str=None, style_: str=None, tabbable: bool=None, tooltip: str=None, touch: dict=None, touchless: bool=None, v_model: Any='!!disabled!!', v_on: str=None, v_slots: list=[], value: Any=None, vertical: bool=None, on_v_model: typing.Callable[[Any], Any]=None) -> ValueElement[ipyvuetify.generated.Window, Any]:
    """
    :param tabbable: Is widget tabbable?
    :param tooltip: A tooltip caption.
    """
    ...