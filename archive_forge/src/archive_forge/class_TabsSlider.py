from traitlets import (
from .VuetifyWidget import VuetifyWidget
class TabsSlider(VuetifyWidget):
    _model_name = Unicode('TabsSliderModel').tag(sync=True)
    color = Unicode(None, allow_none=True).tag(sync=True)