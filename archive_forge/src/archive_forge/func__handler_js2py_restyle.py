import ipywidgets as widgets
from traitlets import List, Unicode, Dict, observe, Integer
from .basedatatypes import BaseFigure, BasePlotlyType
from .callbacks import BoxSelector, LassoSelector, InputDeviceState, Points
from .serializers import custom_serializers
from .version import __frontend_version__
@observe('_js2py_restyle')
def _handler_js2py_restyle(self, change):
    """
        Process Plotly.restyle message from the frontend
        """
    restyle_msg = change['new']
    if not restyle_msg:
        self._js2py_restyle = None
        return
    style_data = restyle_msg['style_data']
    style_traces = restyle_msg['style_traces']
    source_view_id = restyle_msg['source_view_id']
    self.plotly_restyle(restyle_data=style_data, trace_indexes=style_traces, source_view_id=source_view_id)
    self._js2py_restyle = None