import abc
import collections
import inspect
import sys
import uuid
import random
from .._utils import patch_collections_abc, stringify_id, OrderedSet
def _traverse_ids(self):
    """Yield components with IDs in the tree of children."""
    for t in self._traverse():
        if isinstance(t, Component) and getattr(t, 'id', None) is not None:
            yield t