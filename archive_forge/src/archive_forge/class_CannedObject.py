import copy
import pickle
import sys
import typing
import warnings
from types import FunctionType
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
class CannedObject:
    """A canned object."""

    def __init__(self, obj, keys=None, hook=None):
        """can an object for safe pickling

        Parameters
        ----------
        obj
            The object to be canned
        keys : list (optional)
            list of attribute names that will be explicitly canned / uncanned
        hook : callable (optional)
            An optional extra callable,
            which can do additional processing of the uncanned object.

        Notes
        -----
        large data may be offloaded into the buffers list,
        used for zero-copy transfers.
        """
        self.keys = keys or []
        self.obj = copy.copy(obj)
        self.hook = can(hook)
        for key in keys:
            setattr(self.obj, key, can(getattr(obj, key)))
        self.buffers = []

    def get_object(self, g=None):
        """Get an object."""
        if g is None:
            g = {}
        obj = self.obj
        for key in self.keys:
            setattr(obj, key, uncan(getattr(obj, key), g))
        if self.hook:
            self.hook = uncan(self.hook, g)
            self.hook(obj, g)
        return self.obj