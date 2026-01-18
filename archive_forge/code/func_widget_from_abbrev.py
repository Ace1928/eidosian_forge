from collections.abc import Iterable, Mapping
from inspect import signature, Parameter
from inspect import getcallargs
from inspect import getfullargspec as check_argspec
import sys
from IPython import get_ipython
from . import (Widget, ValueWidget, Text,
from IPython.display import display, clear_output
from traitlets import HasTraits, Any, Unicode, observe
from numbers import Real, Integral
from warnings import warn
@classmethod
def widget_from_abbrev(cls, abbrev, default=empty):
    """Build a ValueWidget instance given an abbreviation or Widget."""
    if isinstance(abbrev, ValueWidget) or isinstance(abbrev, fixed):
        return abbrev
    if isinstance(abbrev, tuple):
        widget = cls.widget_from_tuple(abbrev)
        if default is not empty:
            try:
                widget.value = default
            except Exception:
                pass
        return widget
    widget = cls.widget_from_single_value(abbrev)
    if widget is not None:
        return widget
    if isinstance(abbrev, Iterable):
        widget = cls.widget_from_iterable(abbrev)
        if default is not empty:
            try:
                widget.value = default
            except Exception:
                pass
        return widget
    return None