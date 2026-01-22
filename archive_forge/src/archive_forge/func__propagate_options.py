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
@observe('options')
def _propagate_options(self, change):
    """Select the first range"""
    options = self._options_full
    self.set_trait('_options_labels', tuple((i[0] for i in options)))
    self._options_values = tuple((i[1] for i in options))
    if self._initializing_traits_ is not True:
        self.index = (0, 0)