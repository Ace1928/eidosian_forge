import typing
from typing import Any, Dict, Union
import ipyvue
import ipywidgets
import reacton
from reacton import ipywidgets as w
from reacton.core import Element, ValueElement
from reacton.utils import implements
import ipyvuetify
def _DataIterator(attributes: dict={}, children: list=[], class_: str=None, dark: bool=None, disable_filtering: bool=None, disable_pagination: bool=None, disable_sort: bool=None, expanded: list=[], footer_props: dict=None, group_by: typing.Union[str, list]=None, group_desc: typing.Union[bool, list]=None, hide_default_footer: bool=None, item_key: str=None, items: list=[], items_per_page: float=None, layout: Union[Dict[str, Any], Element[ipywidgets.widgets.widget_layout.Layout]]={}, light: bool=None, loading: typing.Union[bool, str]=None, loading_text: str=None, locale: str=None, mobile_breakpoint: typing.Union[float, str]=None, multi_sort: bool=None, must_sort: bool=None, no_data_text: str=None, no_results_text: str=None, options: dict=None, page: float=None, search: str=None, selectable_key: str=None, server_items_length: float=None, single_expand: bool=None, single_select: bool=None, slot: str=None, sort_by: typing.Union[str, list]=None, sort_desc: typing.Union[bool, list]=None, style_: str=None, tabbable: bool=None, tooltip: str=None, v_model: Any='!!disabled!!', v_on: str=None, v_slots: list=[], value: list=[], on_v_model: typing.Callable[[Any], Any]=None) -> ValueElement[ipyvuetify.generated.DataIterator, Any]:
    """
    :param tabbable: Is widget tabbable?
    :param tooltip: A tooltip caption.
    """
    ...