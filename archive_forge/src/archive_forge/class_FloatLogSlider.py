from traitlets import (
from .widget_description import DescriptionWidget
from .trait_types import InstanceDict, NumberFormat
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .widget_core import CoreWidget
from .widget_int import ProgressStyle, SliderStyle
@register
class FloatLogSlider(_BoundedLogFloat):
    """ Slider/trackbar of logarithmic floating values with the specified range.

    Parameters
    ----------
    value : float
        position of the slider
    base : float
        base of the logarithmic scale. Default is 10
    min : float
        minimal position of the slider in log scale, i.e., actual minimum is base ** min
    max : float
        maximal position of the slider in log scale, i.e., actual maximum is base ** max
    step : float
        step of the trackbar, denotes steps for the exponent, not the actual value
    description : str
        name of the slider
    orientation : {'horizontal', 'vertical'}
        default is 'horizontal', orientation of the slider
    readout : {True, False}
        default is True, display the current value of the slider next to it
    behavior : str
        slider handle and connector dragging behavior. Default is 'drag-tap'.
    readout_format : str
        default is '.3g', specifier for the format function used to represent
        slider value for human consumption, modeled after Python 3's format
        specification mini-language (PEP 3101).
    """
    _view_name = Unicode('FloatLogSliderView').tag(sync=True)
    _model_name = Unicode('FloatLogSliderModel').tag(sync=True)
    step = CFloat(0.1, allow_none=True, help='Minimum step in the exponent to increment the value').tag(sync=True)
    orientation = CaselessStrEnum(values=['horizontal', 'vertical'], default_value='horizontal', help='Vertical or horizontal.').tag(sync=True)
    readout = Bool(True, help='Display the current value of the slider next to it.').tag(sync=True)
    readout_format = NumberFormat('.3g', help='Format for the readout').tag(sync=True)
    continuous_update = Bool(True, help='Update the value of the widget as the user is holding the slider.').tag(sync=True)
    disabled = Bool(False, help='Enable or disable user changes').tag(sync=True)
    base = CFloat(10.0, help='Base for the logarithm').tag(sync=True)
    style = InstanceDict(SliderStyle).tag(sync=True, **widget_serialization)
    behavior = CaselessStrEnum(values=['drag-tap', 'drag-snap', 'tap', 'drag', 'snap'], default_value='drag-tap', help='Slider dragging behavior.').tag(sync=True)