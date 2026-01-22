from ipywidgets import Widget, widget_serialization
from traitlets import Unicode
from .._package import npm_pkg_name
from .._version import EXTENSION_SPEC_VERSION
def _on_ret_val(self, method_name, ret_val):
    """Message callback used internally for logging exec returns"""
    self.log.info('%s() -> %s' % (method_name, ret_val))