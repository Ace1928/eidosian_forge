import copy
import pickle
import sys
import typing
import warnings
from types import FunctionType
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
def uncan(obj, g=None):
    """invert canning"""
    import_needed = False
    for cls, uncanner in uncan_map.items():
        if isinstance(cls, str):
            import_needed = True
            break
        if isinstance(obj, cls):
            return uncanner(obj, g)
    if import_needed:
        _import_mapping(uncan_map, _original_uncan_map)
        return uncan(obj, g)
    return obj