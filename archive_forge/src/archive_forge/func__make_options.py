from collections.abc import Iterable, Mapping
from itertools import chain
from .widget_description import DescriptionWidget, DescriptionStyle
from .valuewidget import ValueWidget
from .widget_core import CoreWidget
from .widget_style import Style
from .trait_types import InstanceDict, TypedTuple
from .widget import register, widget_serialization
from .widget_int import SliderStyle
from .docutils import doc_subst
from traitlets import (Unicode, Bool, Int, Any, Dict, TraitError, CaselessStrEnum,
def _make_options(x):
    """Standardize the options tuple format.

    The returned tuple should be in the format (('label', value), ('label', value), ...).

    The input can be
    * an iterable of (label, value) pairs
    * an iterable of values, and labels will be generated
    * a Mapping between labels and values
    """
    if isinstance(x, Mapping):
        x = x.items()
    xlist = tuple(x)
    if all((isinstance(i, (list, tuple)) and len(i) == 2 for i in xlist)):
        return tuple(((str(k), v) for k, v in xlist))
    return tuple(((str(i), i) for i in xlist))