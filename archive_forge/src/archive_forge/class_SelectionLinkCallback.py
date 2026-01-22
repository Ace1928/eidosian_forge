import numpy as np
from bokeh.models import CustomJS, Toolbar
from bokeh.models.tools import RangeTool
from ...core.util import isscalar
from ..links import (
from ..plot import GenericElementPlot, GenericOverlayPlot
class SelectionLinkCallback(LinkCallback):
    source_model = 'selected'
    target_model = 'selected'
    on_source_changes = ['indices']
    on_target_changes = ['indices']
    source_handles = ['cds']
    target_handles = ['cds']
    source_code = '\n    target_selected.indices = source_selected.indices\n    target_cds.properties.selected.change.emit()\n    '
    target_code = '\n    source_selected.indices = target_selected.indices\n    source_cds.properties.selected.change.emit()\n    '