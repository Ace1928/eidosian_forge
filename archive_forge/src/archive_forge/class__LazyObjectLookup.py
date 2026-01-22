import os.path
import warnings
import weakref
from collections import ChainMap, Counter, OrderedDict, defaultdict
from collections.abc import Mapping
import h5py
import numpy as np
from packaging import version
from . import __version__
from .attrs import Attributes
from .dimensions import Dimension, Dimensions
from .utils import Frozen
class _LazyObjectLookup(Mapping):

    def __init__(self, parent, object_cls):
        self._parent_ref = weakref.ref(parent)
        self._object_cls = object_cls
        self._objects = OrderedDict()

    @property
    def _parent(self):
        return self._parent_ref()

    def __setitem__(self, name, obj):
        self._objects[name] = obj

    def add(self, name):
        self._objects[name] = None

    def __iter__(self):
        for name in self._objects:
            yield name.replace('_nc4_non_coord_', '')

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, key):
        if key not in self._objects and '_nc4_non_coord_' + key in self._objects:
            key = '_nc4_non_coord_' + key
        if self._objects[key] is not None:
            return self._objects[key]
        else:
            self._objects[key] = self._object_cls(self._parent, key)
            return self._objects[key]