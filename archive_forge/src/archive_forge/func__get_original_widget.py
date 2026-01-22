from __future__ import annotations
import typing
import warnings
from urwid.canvas import CompositeCanvas
from .widget import Widget, WidgetError, WidgetWarning, delegate_to_widget_mixin
def _get_original_widget(self) -> WrappedWidget:
    warnings.warn(f'Method `{self.__class__.__name__}._get_original_widget` is deprecated, please use property `{self.__class__.__name__}.original_widget`', DeprecationWarning, stacklevel=2)
    return self.original_widget