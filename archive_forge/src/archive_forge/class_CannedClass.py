import copy
import pickle
import sys
import typing
import warnings
from types import FunctionType
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
class CannedClass(CannedObject):
    """A canned class object."""

    def __init__(self, cls):
        """Initialize the can."""
        self._check_type(cls)
        self.name = cls.__name__
        self.old_style = not isinstance(cls, type)
        self._canned_dict = {}
        for k, v in cls.__dict__.items():
            if k not in ('__weakref__', '__dict__'):
                self._canned_dict[k] = can(v)
        mro = [] if self.old_style else cls.mro()
        self.parents = [can(c) for c in mro[1:]]
        self.buffers = []

    def _check_type(self, obj):
        assert isinstance(obj, class_type), 'Not a class type'

    def get_object(self, g=None):
        """Get an object from the can."""
        parents = tuple((uncan(p, g) for p in self.parents))
        return type(self.name, parents, uncan_dict(self._canned_dict, g=g))