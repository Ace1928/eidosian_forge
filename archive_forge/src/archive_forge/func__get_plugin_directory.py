import weakref
from oslo_concurrency import lockutils
from neutron_lib.plugins import constants
def _get_plugin_directory():
    if _PLUGIN_DIRECTORY is None:
        return _create_plugin_directory()
    return _PLUGIN_DIRECTORY