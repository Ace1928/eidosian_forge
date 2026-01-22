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
@register
@doc_subst(_doc_snippets)
class SelectionRangeSlider(_MultipleSelectionNonempty):
    """
    Slider to select multiple contiguous items from a list.

    The index, value, and label attributes contain the start and end of
    the selection range, not all items in the range.

    Parameters
    ----------
    {multiple_selection_params}

    {slider_params}
    """
    _view_name = Unicode('SelectionRangeSliderView').tag(sync=True)
    _model_name = Unicode('SelectionRangeSliderModel').tag(sync=True)
    value = Tuple(help='Min and max selected values')
    label = Tuple(help='Min and max selected labels')
    index = Tuple((0, 0), help='Min and max selected indices').tag(sync=True)

    @observe('options')
    def _propagate_options(self, change):
        """Select the first range"""
        options = self._options_full
        self.set_trait('_options_labels', tuple((i[0] for i in options)))
        self._options_values = tuple((i[1] for i in options))
        if self._initializing_traits_ is not True:
            self.index = (0, 0)

    @validate('index')
    def _validate_index(self, proposal):
        """Make sure we have two indices and check the range of each proposed index."""
        if len(proposal.value) != 2:
            raise TraitError('Invalid selection: index must have two values, but is %r' % (proposal.value,))
        if all((0 <= i < len(self._options_labels) for i in proposal.value)):
            return proposal.value
        else:
            raise TraitError('Invalid selection: index out of bounds: %s' % (proposal.value,))
    orientation = CaselessStrEnum(values=['horizontal', 'vertical'], default_value='horizontal', help='Vertical or horizontal.').tag(sync=True)
    readout = Bool(True, help='Display the current selected label next to the slider').tag(sync=True)
    continuous_update = Bool(True, help='Update the value of the widget as the user is holding the slider.').tag(sync=True)
    style = InstanceDict(SliderStyle).tag(sync=True, **widget_serialization)
    behavior = CaselessStrEnum(values=['drag-tap', 'drag-snap', 'tap', 'drag', 'snap'], default_value='drag-tap', help='Slider dragging behavior.').tag(sync=True)