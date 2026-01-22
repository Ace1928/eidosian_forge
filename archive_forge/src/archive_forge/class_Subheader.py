from traitlets import (
from .VuetifyWidget import VuetifyWidget
class Subheader(VuetifyWidget):
    _model_name = Unicode('SubheaderModel').tag(sync=True)
    dark = Bool(None, allow_none=True).tag(sync=True)
    inset = Bool(None, allow_none=True).tag(sync=True)
    light = Bool(None, allow_none=True).tag(sync=True)