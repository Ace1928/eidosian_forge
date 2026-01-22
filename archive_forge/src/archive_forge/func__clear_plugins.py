import os.path
import warnings
from configparser import ConfigParser
from glob import glob
from .collection import imread_collection_wrapper
def _clear_plugins():
    """Clear the plugin state to the default, i.e., where no plugins are loaded"""
    global plugin_store
    plugin_store = {'imread': [], 'imsave': [], 'imshow': [], 'imread_collection': [], 'imshow_collection': [], '_app_show': []}