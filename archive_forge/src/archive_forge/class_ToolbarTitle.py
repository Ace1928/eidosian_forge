from traitlets import (
from .VuetifyWidget import VuetifyWidget
class ToolbarTitle(VuetifyWidget):
    _model_name = Unicode('ToolbarTitleModel').tag(sync=True)