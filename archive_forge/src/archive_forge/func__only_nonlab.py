import json
import os.path as osp
from itertools import filterfalse
from .jlpmapp import HERE
def _only_nonlab(collection):
    """Filter a dict/sequence to remove all lab packages

    This is useful to take the default values of e.g. singletons and filter
    away the '@jupyterlab/' namespace packages, but leave any others (e.g.
    lumino and react).
    """
    if isinstance(collection, dict):
        return {k: v for k, v in collection.items() if not _is_lab_package(k)}
    elif isinstance(collection, (list, tuple)):
        return list(filterfalse(_is_lab_package, collection))
    msg = 'collection arg should be either dict or list/tuple'
    raise TypeError(msg)