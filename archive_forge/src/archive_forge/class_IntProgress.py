from .widget_description import DescriptionWidget, DescriptionStyle
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .widget_core import CoreWidget
from traitlets import Instance
from .trait_types import Color, InstanceDict, NumberFormat
from traitlets import (
@register
@_bounded_int_doc
class IntProgress(_BoundedInt):
    """Progress bar that represents an integer bounded from above and below.
    """
    _view_name = Unicode('ProgressView').tag(sync=True)
    _model_name = Unicode('IntProgressModel').tag(sync=True)
    orientation = CaselessStrEnum(values=['horizontal', 'vertical'], default_value='horizontal', help='Vertical or horizontal.').tag(sync=True)
    bar_style = CaselessStrEnum(values=['success', 'info', 'warning', 'danger', ''], default_value='', help='Use a predefined styling for the progress bar.').tag(sync=True)
    style = InstanceDict(ProgressStyle).tag(sync=True, **widget_serialization)