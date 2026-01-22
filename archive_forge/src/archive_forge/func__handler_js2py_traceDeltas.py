import ipywidgets as widgets
from traitlets import List, Unicode, Dict, observe, Integer
from .basedatatypes import BaseFigure, BasePlotlyType
from .callbacks import BoxSelector, LassoSelector, InputDeviceState, Points
from .serializers import custom_serializers
from .version import __frontend_version__
@observe('_js2py_traceDeltas')
def _handler_js2py_traceDeltas(self, change):
    """
        Process trace deltas message from the frontend
        """
    msg_data = change['new']
    if not msg_data:
        self._js2py_traceDeltas = None
        return
    trace_deltas = msg_data['trace_deltas']
    trace_edit_id = msg_data['trace_edit_id']
    if trace_edit_id == self._last_trace_edit_id:
        for delta in trace_deltas:
            trace_uid = delta['uid']
            trace_uids = [trace.uid for trace in self.data]
            trace_index = trace_uids.index(trace_uid)
            uid_trace = self.data[trace_index]
            delta_transform = BaseFigureWidget._transform_data(uid_trace._prop_defaults, delta)
            remove_props = self._remove_overlapping_props(uid_trace._props, uid_trace._prop_defaults)
            if remove_props:
                remove_trace_props_msg = {'remove_trace': trace_index, 'remove_props': remove_props}
                self._py2js_removeTraceProps = remove_trace_props_msg
                self._py2js_removeTraceProps = None
            self._dispatch_trace_change_callbacks(delta_transform, [trace_index])
        self._trace_edit_in_process = False
        if not self._layout_edit_in_process:
            while self._waiting_edit_callbacks:
                self._waiting_edit_callbacks.pop()()
    self._js2py_traceDeltas = None