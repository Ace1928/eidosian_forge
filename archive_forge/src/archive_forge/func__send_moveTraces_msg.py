import ipywidgets as widgets
from traitlets import List, Unicode, Dict, observe, Integer
from .basedatatypes import BaseFigure, BasePlotlyType
from .callbacks import BoxSelector, LassoSelector, InputDeviceState, Points
from .serializers import custom_serializers
from .version import __frontend_version__
def _send_moveTraces_msg(self, current_inds, new_inds):
    """
        Send Plotly.moveTraces message to the frontend

        Parameters
        ----------
        current_inds : list[int]
            List of current trace indexes
        new_inds : list[int]
            List of new trace indexes
        """
    move_msg = {'current_trace_inds': current_inds, 'new_trace_inds': new_inds}
    self._py2js_moveTraces = move_msg
    self._py2js_moveTraces = None