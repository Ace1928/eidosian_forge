import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def tree_apply(f, tree, is_leaf=is_not_container):
    """Apply ``f`` to all leaves in ``tree``, no new pytree is built.

    Parameters
    ----------
    f : callable
        A function to apply to all leaves in ``tree``.
    tree : pytree
        A nested sequence of tuples, lists, dicts and other objects.
    is_leaf : callable
        A function to determine if an object is a leaf, ``f`` is only applied
        to objects for which ``is_leaf(x)`` returns ``True``.
    """
    if is_leaf(tree):
        f(tree)
        return
    try:
        TREE_APPLIER_CACHE[tree.__class__](f, tree, is_leaf)
    except KeyError:
        for cls, applier in reversed(TREE_APPLY_REGISTRY.items()):
            if isinstance(tree, cls):
                break
        else:
            applier = nothing
        TREE_APPLIER_CACHE[tree.__class__] = applier
        applier(f, tree, is_leaf)