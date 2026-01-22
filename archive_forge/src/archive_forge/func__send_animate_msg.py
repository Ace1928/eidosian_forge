import ipywidgets as widgets
from traitlets import List, Unicode, Dict, observe, Integer
from .basedatatypes import BaseFigure, BasePlotlyType
from .callbacks import BoxSelector, LassoSelector, InputDeviceState, Points
from .serializers import custom_serializers
from .version import __frontend_version__
def _send_animate_msg(self, styles_data, relayout_data, trace_indexes, animation_opts):
    """
        Send Plotly.update message to the frontend

        Note: there is no source_view_id parameter because animations
        triggered by the fontend are not currently supported

        Parameters
        ----------
        styles_data : list[dict]
            Plotly.animate styles data
        relayout_data : dict
            Plotly.animate relayout data
        trace_indexes : list[int]
            List of trace indexes that the animate operation applies to
        """
    trace_indexes = self._normalize_trace_indexes(trace_indexes)
    trace_edit_id = self._last_trace_edit_id + 1
    self._last_trace_edit_id = trace_edit_id
    self._trace_edit_in_process = True
    layout_edit_id = self._last_layout_edit_id + 1
    self._last_layout_edit_id = layout_edit_id
    self._layout_edit_in_process = True
    animate_msg = {'style_data': styles_data, 'layout_data': relayout_data, 'style_traces': trace_indexes, 'animation_opts': animation_opts, 'trace_edit_id': trace_edit_id, 'layout_edit_id': layout_edit_id, 'source_view_id': None}
    self._py2js_animate = animate_msg
    self._py2js_animate = None