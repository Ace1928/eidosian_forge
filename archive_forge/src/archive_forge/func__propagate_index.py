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
@observe('index')
def _propagate_index(self, change):
    """Propagate changes in index to the value and label properties"""
    label = tuple((self._options_labels[i] for i in change.new))
    value = tuple((self._options_values[i] for i in change.new))
    if self.label != label:
        self.label = label
    if self.value != value:
        self.value = value