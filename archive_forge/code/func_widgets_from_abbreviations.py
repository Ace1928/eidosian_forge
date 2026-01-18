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
def widgets_from_abbreviations(self, seq):
    """Given a sequence of (name, abbrev, default) tuples, return a sequence of Widgets."""
    result = []
    for name, abbrev, default in seq:
        if isinstance(abbrev, Widget) and (not isinstance(abbrev, ValueWidget)):
            raise TypeError('{!r} is not a ValueWidget'.format(abbrev))
        widget = self.widget_from_abbrev(abbrev, default)
        if widget is None:
            raise ValueError('{!r} cannot be transformed to a widget'.format(abbrev))
        if not hasattr(widget, 'description') or not widget.description:
            widget.description = name
        widget._kwarg = name
        result.append(widget)
    return result