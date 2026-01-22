import os.path
import warnings
from configparser import ConfigParser
from glob import glob
from .collection import imread_collection_wrapper
def _inject_imread_collection_if_needed(module):
    """Add `imread_collection` to module if not already present."""
    if not hasattr(module, 'imread_collection') and hasattr(module, 'imread'):
        imread = getattr(module, 'imread')
        func = imread_collection_wrapper(imread)
        setattr(module, 'imread_collection', func)