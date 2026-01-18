import copy
import pickle
import sys
import typing
import warnings
from types import FunctionType
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
def uncan_dict(obj, g=None):
    """Uncan a dict object."""
    if istype(obj, dict):
        newobj = {}
        for k, v in obj.items():
            newobj[k] = uncan(v, g)
        return newobj
    return obj