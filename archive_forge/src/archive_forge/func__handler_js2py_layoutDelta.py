import ipywidgets as widgets
from traitlets import List, Unicode, Dict, observe, Integer
from .basedatatypes import BaseFigure, BasePlotlyType
from .callbacks import BoxSelector, LassoSelector, InputDeviceState, Points
from .serializers import custom_serializers
from .version import __frontend_version__
@observe('_js2py_layoutDelta')
def _handler_js2py_layoutDelta(self, change):
    """
        Process layout delta message from the frontend
        """
    msg_data = change['new']
    if not msg_data:
        self._js2py_layoutDelta = None
        return
    layout_delta = msg_data['layout_delta']
    layout_edit_id = msg_data['layout_edit_id']
    if layout_edit_id == self._last_layout_edit_id:
        delta_transform = BaseFigureWidget._transform_data(self._layout_defaults, layout_delta)
        removed_props = self._remove_overlapping_props(self._layout, self._layout_defaults)
        if removed_props:
            remove_props_msg = {'remove_props': removed_props}
            self._py2js_removeLayoutProps = remove_props_msg
            self._py2js_removeLayoutProps = None
        for proppath in delta_transform:
            prop = proppath[0]
            match = self.layout._subplot_re_match(prop)
            if match and prop not in self.layout:
                self.layout[prop] = {}
        self._dispatch_layout_change_callbacks(delta_transform)
        self._layout_edit_in_process = False
        if not self._trace_edit_in_process:
            while self._waiting_edit_callbacks:
                self._waiting_edit_callbacks.pop()()
    self._js2py_layoutDelta = None