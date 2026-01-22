import contextlib
import copy
import weakref
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.trackable import base
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
class ResourceTracker:
    """An object that tracks a list of resources."""
    __slots__ = ['_resources']

    def __init__(self):
        self._resources = []

    @property
    def resources(self):
        return self._resources

    def add_resource(self, resource):
        self._resources.append(resource)