import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def tree_register_container(cls, mapper, iterator, applier):
    """Register a new container type for use with ``tree_map`` and
    ``tree_apply``.

    Parameters
    ----------
    cls : type
        The container type to register.
    mapper : callable
        A function that takes ``f``, ``tree`` and ``is_leaf`` and returns a new
        tree of type ``cls`` with ``f`` applied to all leaves.
    applier : callable
        A function that takes ``f``, ``tree`` and ``is_leaf`` and applies ``f``
        to all leaves in ``tree``.
    """
    TREE_MAP_REGISTRY[cls] = mapper
    TREE_ITER_REGISTRY[cls] = iterator
    TREE_APPLY_REGISTRY[cls] = applier