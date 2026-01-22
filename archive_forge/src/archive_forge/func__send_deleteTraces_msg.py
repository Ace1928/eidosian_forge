import ipywidgets as widgets
from traitlets import List, Unicode, Dict, observe, Integer
from .basedatatypes import BaseFigure, BasePlotlyType
from .callbacks import BoxSelector, LassoSelector, InputDeviceState, Points
from .serializers import custom_serializers
from .version import __frontend_version__
def _send_deleteTraces_msg(self, delete_inds):
    """
        Send Plotly.deleteTraces message to the frontend

        Parameters
        ----------
        delete_inds : list[int]
            List of trace indexes of traces to delete
        """
    trace_edit_id = self._last_trace_edit_id + 1
    self._last_trace_edit_id = trace_edit_id
    self._trace_edit_in_process = True
    layout_edit_id = self._last_layout_edit_id + 1
    self._last_layout_edit_id = layout_edit_id
    self._layout_edit_in_process = True
    delete_msg = {'delete_inds': delete_inds, 'layout_edit_id': layout_edit_id, 'trace_edit_id': trace_edit_id}
    self._py2js_deleteTraces = delete_msg
    self._py2js_deleteTraces = None