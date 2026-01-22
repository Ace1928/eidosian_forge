from traitlets import (
from .VuetifyWidget import VuetifyWidget
class Breadcrumbs(VuetifyWidget):
    _model_name = Unicode('BreadcrumbsModel').tag(sync=True)
    dark = Bool(None, allow_none=True).tag(sync=True)
    divider = Unicode(None, allow_none=True).tag(sync=True)
    items = List(Any(), default_value=None, allow_none=True).tag(sync=True)
    large = Bool(None, allow_none=True).tag(sync=True)
    light = Bool(None, allow_none=True).tag(sync=True)