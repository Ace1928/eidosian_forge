import copy
import pickle
import sys
import typing
import warnings
from types import FunctionType
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
def _import_mapping(mapping, original=None):
    """import any string-keys in a type mapping"""
    log = get_logger()
    log.debug('Importing canning map')
    for key, _ in list(mapping.items()):
        if isinstance(key, str):
            try:
                cls = import_item(key)
            except Exception:
                if original and key not in original:
                    log.error('canning class not importable: %r', key, exc_info=True)
                mapping.pop(key)
            else:
                mapping[cls] = mapping.pop(key)