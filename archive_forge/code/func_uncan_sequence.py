import copy
import pickle
import sys
import typing
import warnings
from types import FunctionType
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
def uncan_sequence(obj, g=None):
    """Uncan a sequence."""
    if istype(obj, sequence_types):
        t = type(obj)
        return t([uncan(i, g) for i in obj])
    return obj