from traitlets import (
from .VuetifyWidget import VuetifyWidget
class Parallax(VuetifyWidget):
    _model_name = Unicode('ParallaxModel').tag(sync=True)
    alt = Unicode(None, allow_none=True).tag(sync=True)
    height = Union([Unicode(), Float()], default_value=None, allow_none=True).tag(sync=True)
    src = Unicode(None, allow_none=True).tag(sync=True)