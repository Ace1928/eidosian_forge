import os.path
import warnings
from configparser import ConfigParser
from glob import glob
from .collection import imread_collection_wrapper
def _set_plugin(plugin_type, plugin_list):
    for plugin in plugin_list:
        if plugin not in available_plugins:
            continue
        try:
            use_plugin(plugin, kind=plugin_type)
            break
        except (ImportError, RuntimeError, OSError):
            pass