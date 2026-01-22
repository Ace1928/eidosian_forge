import sys
import logging; log = logging.getLogger(__name__)
from types import ModuleType
def add_doc(obj, doc):
    """add docstring to an object"""
    obj.__doc__ = doc