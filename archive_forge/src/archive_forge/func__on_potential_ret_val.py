from ipywidgets import Widget, widget_serialization
from traitlets import Unicode
from .._package import npm_pkg_name
from .._version import EXTENSION_SPEC_VERSION
def _on_potential_ret_val(self, widget, content, buffers):
    """Message callback used internally"""
    if content['type'] == 'exec_three_obj_method_retval':
        self._on_ret_val(content['method_name'], content['ret_val'])