from traitlets import (
from .VuetifyWidget import VuetifyWidget
class ThemeProvider(VuetifyWidget):
    _model_name = Unicode('ThemeProviderModel').tag(sync=True)
    dark = Bool(None, allow_none=True).tag(sync=True)
    light = Bool(None, allow_none=True).tag(sync=True)
    root = Bool(None, allow_none=True).tag(sync=True)