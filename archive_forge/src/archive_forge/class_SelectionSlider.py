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
class SelectionSlider(_SelectionNonempty):
    """
    Slider to select a single item from a list or dictionary.

    Parameters
    ----------
    {selection_params}

    {slider_params}
    """
    _view_name = Unicode('SelectionSliderView').tag(sync=True)
    _model_name = Unicode('SelectionSliderModel').tag(sync=True)
    orientation = CaselessStrEnum(values=['horizontal', 'vertical'], default_value='horizontal', help='Vertical or horizontal.').tag(sync=True)
    readout = Bool(True, help='Display the current selected label next to the slider').tag(sync=True)
    continuous_update = Bool(True, help='Update the value of the widget as the user is holding the slider.').tag(sync=True)
    behavior = CaselessStrEnum(values=['drag-tap', 'drag-snap', 'tap', 'drag', 'snap'], default_value='drag-tap', help='Slider dragging behavior.').tag(sync=True)
    style = InstanceDict(SliderStyle).tag(sync=True, **widget_serialization)