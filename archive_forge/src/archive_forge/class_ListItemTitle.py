from traitlets import (
from .VuetifyWidget import VuetifyWidget
class ListItemTitle(VuetifyWidget):
    _model_name = Unicode('ListItemTitleModel').tag(sync=True)