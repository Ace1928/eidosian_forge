from traitlets import (
from .widget_description import DescriptionWidget
from .trait_types import InstanceDict, NumberFormat
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .widget_core import CoreWidget
from .widget_int import ProgressStyle, SliderStyle
@register
class FloatProgress(_BoundedFloat):
    """ Displays a progress bar.

    Parameters
    -----------
    value : float
        position within the range of the progress bar
    min : float
        minimal position of the slider
    max : float
        maximal position of the slider
    description : str
        name of the progress bar
    orientation : {'horizontal', 'vertical'}
        default is 'horizontal', orientation of the progress bar
    bar_style: {'success', 'info', 'warning', 'danger', ''}
        color of the progress bar, default is '' (blue)
        colors are: 'success'-green, 'info'-light blue, 'warning'-orange, 'danger'-red
    """
    _view_name = Unicode('ProgressView').tag(sync=True)
    _model_name = Unicode('FloatProgressModel').tag(sync=True)
    orientation = CaselessStrEnum(values=['horizontal', 'vertical'], default_value='horizontal', help='Vertical or horizontal.').tag(sync=True)
    bar_style = CaselessStrEnum(values=['success', 'info', 'warning', 'danger', ''], default_value='', allow_none=True, help='Use a predefined styling for the progress bar.').tag(sync=True)
    style = InstanceDict(ProgressStyle).tag(sync=True, **widget_serialization)