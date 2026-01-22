import os.path
import warnings
from configparser import ConfigParser
from glob import glob
from .collection import imread_collection_wrapper
def _load_preferred_plugins():
    io_types = ['imsave', 'imshow', 'imread_collection', 'imshow_collection', 'imread']
    for p_type in io_types:
        _set_plugin(p_type, preferred_plugins['all'])
    plugin_types = (p for p in preferred_plugins.keys() if p != 'all')
    for p_type in plugin_types:
        _set_plugin(p_type, preferred_plugins[p_type])