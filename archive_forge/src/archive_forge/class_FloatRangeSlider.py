from traitlets import (
from .widget_description import DescriptionWidget
from .trait_types import InstanceDict, NumberFormat
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .widget_core import CoreWidget
from .widget_int import ProgressStyle, SliderStyle
@register
class FloatRangeSlider(_BoundedFloatRange):
    """ Slider/trackbar that represents a pair of floats bounded by minimum and maximum value.

    Parameters
    ----------
    value : float tuple
        range of the slider displayed
    min : float
        minimal position of the slider
    max : float
        maximal position of the slider
    step : float
        step of the trackbar
    description : str
        name of the slider
    orientation : {'horizontal', 'vertical'}
        default is 'horizontal'
    readout : {True, False}
        default is True, display the current value of the slider next to it
    behavior : str
        slider handle and connector dragging behavior. Default is 'drag-tap'.
    readout_format : str
        default is '.2f', specifier for the format function used to represent
        slider value for human consumption, modeled after Python 3's format
        specification mini-language (PEP 3101).
    """
    _view_name = Unicode('FloatRangeSliderView').tag(sync=True)
    _model_name = Unicode('FloatRangeSliderModel').tag(sync=True)
    step = CFloat(0.1, allow_none=True, help='Minimum step to increment the value').tag(sync=True)
    orientation = CaselessStrEnum(values=['horizontal', 'vertical'], default_value='horizontal', help='Vertical or horizontal.').tag(sync=True)
    readout = Bool(True, help='Display the current value of the slider next to it.').tag(sync=True)
    readout_format = NumberFormat('.2f', help='Format for the readout').tag(sync=True)
    continuous_update = Bool(True, help='Update the value of the widget as the user is sliding the slider.').tag(sync=True)
    disabled = Bool(False, help='Enable or disable user changes').tag(sync=True)
    style = InstanceDict(SliderStyle).tag(sync=True, **widget_serialization)
    behavior = CaselessStrEnum(values=['drag-tap', 'drag-snap', 'tap', 'drag', 'snap'], default_value='drag-tap', help='Slider dragging behavior.').tag(sync=True)