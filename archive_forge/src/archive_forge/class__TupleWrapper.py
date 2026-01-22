import collections
import copy
import sys
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.trackable import layer_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class _TupleWrapper(TrackableDataStructure, wrapt.ObjectProxy):
    """Trackable wrapper for tuples and namedtuples."""

    def __init__(self, original_wrapped_tuple=()):
        add_dependency = []
        substituted_wrapped_tuple = []
        for element in original_wrapped_tuple:
            if isinstance(element, NoDependency):
                add_dependency.append(False)
            else:
                add_dependency.append(True)
            substituted_wrapped_tuple.append(wrap_or_unwrap(element))
        try:
            fields = original_wrapped_tuple._fields
        except AttributeError:
            is_namedtuple = False
        else:
            is_namedtuple = True
        original_type = type(original_wrapped_tuple)
        self._self_tuple_is_constructable = True
        if is_namedtuple:
            try:
                substituted_wrapped_tuple = original_type(**dict(zip(fields, substituted_wrapped_tuple)))
            except TypeError:
                wrapt.ObjectProxy.__init__(self, original_wrapped_tuple)
                TrackableDataStructure.__init__(self)
                self._self_tuple_is_constructable = False
                return
        else:
            substituted_wrapped_tuple = original_type(substituted_wrapped_tuple)
        wrapt.ObjectProxy.__init__(self, substituted_wrapped_tuple)
        TrackableDataStructure.__init__(self)
        if is_namedtuple:
            for name, should_depend, element in zip(fields, add_dependency, substituted_wrapped_tuple):
                if should_depend:
                    self._track_value(element, name=name)
        for index, (should_depend, element) in enumerate(zip(add_dependency, substituted_wrapped_tuple)):
            if should_depend:
                self._track_value(element, name='%d' % (index,))

    @property
    def _values(self):
        """Collect values for TrackableDataStructure."""
        return self

    def _track_value(self, value, name):
        """Allows storage of non-trackable objects."""
        try:
            value = super()._track_value(value=value, name=name)
        except ValueError:
            value = sticky_attribute_assignment(trackable=self, value=value, name=name)
        return value

    def __repr__(self):
        return '_TupleWrapper(%s)' % (repr(self.__wrapped__),)

    def __hash__(self):
        return hash(self.__wrapped__)

    def __eq__(self, other):
        return self.__wrapped__ == other

    def __copy__(self):
        return _TupleWrapper(copy.copy(self.__wrapped__))

    def __deepcopy__(self, memo):
        return _TupleWrapper(copy.deepcopy(self.__wrapped__, memo))

    def __reduce_ex__(self, protocol):
        return (self.__class__, (self.__wrapped__,))

    def __imul__(self, y):
        """Avoid running self.__wrapped__ *= y, which mutates `self`."""
        return self.__wrapped__ * y

    def __iadd__(self, y):
        """Avoid running self.__wrapped__ += y, which mutates `self`."""
        return self.__wrapped__ + y

    def _trackable_children(self, save_type=base.SaveType.CHECKPOINT, **kwargs):
        if not self._self_tuple_is_constructable:
            raise ValueError(f'Unable to save because the namedtuple {self.__wrapped__} is not constructable from its _fields (i.e. __new__ is overridden). Expected keyword arguments {self.__wrapped__._fields}. If you do not need to save this object, consider wrapping it in a custom object that does not inherit from tuple.')
        return super()._trackable_children(save_type, **kwargs)

    def __getattribute__(self, name):
        if name != '__wrapped__' and hasattr(self.__wrapped__, name):
            return getattr(self.__wrapped__, name)
        if hasattr(type(self), name) and isinstance(getattr(type(self), name), property):
            return object.__getattribute__(self, name)
        else:
            return super().__getattribute__(name)