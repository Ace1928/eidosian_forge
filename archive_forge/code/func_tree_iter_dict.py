import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def tree_iter_dict(tree, is_leaf):
    for v in tree.values():
        yield from tree_iter(v, is_leaf)