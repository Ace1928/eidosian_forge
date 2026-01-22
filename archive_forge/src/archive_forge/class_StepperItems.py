from traitlets import (
from .VuetifyWidget import VuetifyWidget
class StepperItems(VuetifyWidget):
    _model_name = Unicode('StepperItemsModel').tag(sync=True)