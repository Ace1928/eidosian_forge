from traitlets import (
from ipywidgets import DOMWidget, register, widget_serialization
from .scales import Scale, LinearScale
from .interacts import Interaction
from .marks import Mark
from .axes import Axis
from ._version import __frontend_version__
@default('scale_x')
def _default_scale_x(self):
    return LinearScale(min=0, max=1, allow_padding=False)