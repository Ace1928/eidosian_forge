import typing
from typing import Any, Dict, Union
import ipyvue
import ipywidgets
import reacton
from reacton import ipywidgets as w
from reacton.core import Element, ValueElement
from reacton.utils import implements
import ipyvuetify
def _VirtualTable(attributes: dict={}, children: list=[], chunk_size: float=None, class_: str=None, dark: bool=None, dense: bool=None, fixed_header: bool=None, header_height: float=None, height: typing.Union[float, str]=None, items: list=[], layout: Union[Dict[str, Any], Element[ipywidgets.widgets.widget_layout.Layout]]={}, light: bool=None, row_height: float=None, slot: str=None, style_: str=None, tabbable: bool=None, tooltip: str=None, v_model: Any='!!disabled!!', v_on: str=None, v_slots: list=[], on_v_model: typing.Callable[[Any], Any]=None) -> ValueElement[ipyvuetify.generated.VirtualTable, Any]:
    """
    :param tabbable: Is widget tabbable?
    :param tooltip: A tooltip caption.
    """
    ...