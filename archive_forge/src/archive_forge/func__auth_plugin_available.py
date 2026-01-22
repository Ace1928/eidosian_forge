import abc
import stevedore
from keystoneauth1 import exceptions
def _auth_plugin_available(ext):
    """Read the value of available for whether to load this plugin."""
    return ext.obj.available