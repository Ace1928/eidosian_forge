from traitlets import (
from .VuetifyWidget import VuetifyWidget
class SimpleTable(VuetifyWidget):
    _model_name = Unicode('SimpleTableModel').tag(sync=True)
    dark = Bool(None, allow_none=True).tag(sync=True)
    dense = Bool(None, allow_none=True).tag(sync=True)
    fixed_header = Bool(None, allow_none=True).tag(sync=True)
    height = Union([Float(), Unicode()], default_value=None, allow_none=True).tag(sync=True)
    light = Bool(None, allow_none=True).tag(sync=True)