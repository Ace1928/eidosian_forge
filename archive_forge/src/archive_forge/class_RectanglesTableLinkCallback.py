import numpy as np
from bokeh.models import CustomJS, Toolbar
from bokeh.models.tools import RangeTool
from ...core.util import isscalar
from ..links import (
from ..plot import GenericElementPlot, GenericOverlayPlot
class RectanglesTableLinkCallback(DataLinkCallback):
    source_model = 'cds'
    target_model = 'cds'
    source_handles = ['glyph']
    on_source_changes = ['selected', 'data']
    on_target_changes = ['patching']
    source_code = '\n    target_cds.data[columns[0]] = source_cds.data[source_glyph.left.field]\n    target_cds.data[columns[1]] = source_cds.data[source_glyph.bottom.field]\n    target_cds.data[columns[2]] = source_cds.data[source_glyph.right.field]\n    target_cds.data[columns[3]] = source_cds.data[source_glyph.top.field]\n    '
    target_code = "\n    source_cds.data['left'] = target_cds.data[columns[0]]\n    source_cds.data['bottom'] = target_cds.data[columns[1]]\n    source_cds.data['right'] = target_cds.data[columns[2]]\n    source_cds.data['top'] = target_cds.data[columns[3]]\n    "

    def __init__(self, root_model, link, source_plot, target_plot=None):
        DataLinkCallback.__init__(self, root_model, link, source_plot, target_plot)
        LinkCallback.__init__(self, root_model, link, source_plot, target_plot)
        columns = [kd.name for kd in source_plot.current_frame.kdims]
        self.src_cb.args['columns'] = columns
        self.tgt_cb.args['columns'] = columns