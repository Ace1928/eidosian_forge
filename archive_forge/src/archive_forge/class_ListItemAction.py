from traitlets import (
from .VuetifyWidget import VuetifyWidget
class ListItemAction(VuetifyWidget):
    _model_name = Unicode('ListItemActionModel').tag(sync=True)