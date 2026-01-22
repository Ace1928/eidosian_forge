import abc
import collections
import inspect
import sys
import uuid
import random
from .._utils import patch_collections_abc, stringify_id, OrderedSet
def _check_if_has_indexable_children(item):
    if not hasattr(item, 'children') or (not isinstance(item.children, Component) and (not isinstance(item.children, (tuple, MutableSequence)))):
        raise KeyError