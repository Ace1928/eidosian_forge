import weakref
from oslo_concurrency import lockutils
from neutron_lib.plugins import constants
@_synchronized('plugin-directory')
def _create_plugin_directory():
    global _PLUGIN_DIRECTORY
    if _PLUGIN_DIRECTORY is None:
        _PLUGIN_DIRECTORY = _PluginDirectory()
    return _PLUGIN_DIRECTORY