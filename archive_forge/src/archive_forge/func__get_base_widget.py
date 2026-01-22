from __future__ import annotations
import typing
import warnings
from urwid.canvas import CompositeCanvas
from .widget import Widget, WidgetError, WidgetWarning, delegate_to_widget_mixin
def _get_base_widget(self) -> Widget:
    warnings.warn(f'Method `{self.__class__.__name__}._get_base_widget` is deprecated, please use property `{self.__class__.__name__}.base_widget`', DeprecationWarning, stacklevel=2)
    return self.base_widget