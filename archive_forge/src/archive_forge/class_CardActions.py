from traitlets import (
from .VuetifyWidget import VuetifyWidget
class CardActions(VuetifyWidget):
    _model_name = Unicode('CardActionsModel').tag(sync=True)