from traitlets import (
from .VuetifyWidget import VuetifyWidget
class CardTitle(VuetifyWidget):
    _model_name = Unicode('CardTitleModel').tag(sync=True)