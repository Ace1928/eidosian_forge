from traitlets import (
from ipywidgets import DOMWidget, register, widget_serialization
from .scales import Scale, LinearScale
from .interacts import Interaction
from .marks import Mark
from .axes import Axis
from ._version import __frontend_version__
@validate('min_aspect_ratio', 'max_aspect_ratio')
def _validate_aspect_ratio(self, proposal):
    value = proposal['value']
    if proposal['trait'].name == 'min_aspect_ratio' and value > self.max_aspect_ratio:
        raise TraitError('setting min_aspect_ratio > max_aspect_ratio')
    if proposal['trait'].name == 'max_aspect_ratio' and value < self.min_aspect_ratio:
        raise TraitError('setting max_aspect_ratio < min_aspect_ratio')
    return value