import typing
from typing import Any, Dict, Union
import ipyvue
import ipywidgets
import reacton
from reacton import ipywidgets as w
from reacton.core import Element, ValueElement
from reacton.utils import implements
import ipyvuetify
def _WindowItem(active_class: str=None, attributes: dict={}, children: list=[], class_: str=None, disabled: bool=None, eager: bool=None, layout: Union[Dict[str, Any], Element[ipywidgets.widgets.widget_layout.Layout]]={}, reverse_transition: typing.Union[bool, str]=None, slot: str=None, style_: str=None, tabbable: bool=None, tooltip: str=None, transition: typing.Union[bool, str]=None, v_model: Any='!!disabled!!', v_on: str=None, v_slots: list=[], value: Any=None, on_v_model: typing.Callable[[Any], Any]=None) -> ValueElement[ipyvuetify.generated.WindowItem, Any]:
    """
    :param tabbable: Is widget tabbable?
    :param tooltip: A tooltip caption.
    """
    ...