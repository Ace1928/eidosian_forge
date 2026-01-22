from traitlets import (
from .VuetifyWidget import VuetifyWidget
class DataTableHeader(VuetifyWidget):
    _model_name = Unicode('DataTableHeaderModel').tag(sync=True)
    mobile = Bool(None, allow_none=True).tag(sync=True)