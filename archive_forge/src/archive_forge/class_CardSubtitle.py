from traitlets import (
from .VuetifyWidget import VuetifyWidget
class CardSubtitle(VuetifyWidget):
    _model_name = Unicode('CardSubtitleModel').tag(sync=True)