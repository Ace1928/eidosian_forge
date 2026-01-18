import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def tree_map_dict(f, tree, is_leaf):
    return {k: tree_map(f, v, is_leaf) for k, v in tree.items()}