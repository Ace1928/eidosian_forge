import collections
import weakref
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.trackable import constants
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.tracking.TrackableReference', v1=[])
class TrackableReference(object):
    """A named reference to a trackable object for use with the `Trackable` class.

  These references mark named `Trackable` dependencies of a `Trackable` object
  and should be created when overriding `Trackable._checkpoint_dependencies`.

  Attributes:
    name: The local name for this dependency.
    ref: The `Trackable` object being referenced.
  """
    __slots__ = ('_name', '_ref')

    def __init__(self, name, ref):
        self._name = name
        self._ref = ref

    @property
    def name(self):
        return self._name

    @property
    def ref(self):
        return self._ref

    def __iter__(self):
        yield self.name
        yield self.ref

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, ref={self.ref})'

    def __eq__(self, o):
        if isinstance(o, tuple):
            return (self.name, self.ref) == o
        elif isinstance(o, TrackableReference):
            return self.name == o.name and self.ref == o.ref
        else:
            return False