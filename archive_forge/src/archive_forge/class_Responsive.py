from traitlets import (
from .VuetifyWidget import VuetifyWidget
class Responsive(VuetifyWidget):
    _model_name = Unicode('ResponsiveModel').tag(sync=True)
    aspect_ratio = Union([Unicode(), Float()], default_value=None, allow_none=True).tag(sync=True)
    height = Union([Float(), Unicode()], default_value=None, allow_none=True).tag(sync=True)
    max_height = Union([Float(), Unicode()], default_value=None, allow_none=True).tag(sync=True)
    max_width = Union([Float(), Unicode()], default_value=None, allow_none=True).tag(sync=True)
    min_height = Union([Float(), Unicode()], default_value=None, allow_none=True).tag(sync=True)
    min_width = Union([Float(), Unicode()], default_value=None, allow_none=True).tag(sync=True)
    width = Union([Float(), Unicode()], default_value=None, allow_none=True).tag(sync=True)