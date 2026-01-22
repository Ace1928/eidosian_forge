import os.path
import warnings
from configparser import ConfigParser
from glob import glob
from .collection import imread_collection_wrapper
def _parse_config_file(filename):
    """Return plugin name and meta-data dict from plugin config file."""
    parser = ConfigParser()
    parser.read(filename)
    name = parser.sections()[0]
    meta_data = {}
    for opt in parser.options(name):
        meta_data[opt] = parser.get(name, opt)
    return (name, meta_data)