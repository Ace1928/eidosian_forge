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
class List(TrackableDataStructure, collections_abc.Sequence):
    """An append-only sequence type which is trackable.

  Maintains checkpoint dependencies on its contents (which must also be
  trackable), and forwards any `Layer` metadata such as updates and losses.

  Note that `List` is purely a container. It lets a `tf.keras.Model` or
  other trackable object know about its contents, but does not call any
  `Layer` instances which are added to it. To indicate a sequence of `Layer`
  instances which should be called sequentially, use `tf.keras.Sequential`.

  Example usage:
  ```python
  class HasList(tf.keras.Model):

    def __init__(self):
      super().__init__()
      self.layer_list = List([layers.Dense(3)])
      self.layer_list.append(layers.Dense(4))

    def call(self, x):
      aggregation = 0.
      for l in self.layer_list:
        x = l(x)
        aggregation += tf.reduce_sum(x)
      return aggregation
  ```

  This kind of wrapping is necessary because `Trackable` objects do not
  (yet) deeply inspect regular Python data structures, so for example assigning
  a regular list (`self.layer_list = [layers.Dense(3)]`) does not create a
  checkpoint dependency and does not add the `Layer` instance's weights to its
  parent `Model`.
  """

    def __init__(self, *args, **kwargs):
        """Construct a new sequence. Arguments are passed to `list()`."""
        super().__init__()
        self._storage = self._make_storage(*args, **kwargs)
        for index, element in enumerate(self._storage):
            self._storage[index] = self._track_value(element, name=self._name_element(index))

    def copy(self):
        return type(self)(copy.copy(self._storage))

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return type(self)(copy.deepcopy(self._storage, memo))

    def _make_storage(self, *args, **kwargs):
        """Determines the backing storage (overridden in subclasses)."""
        return list(*args, **kwargs)

    def _name_element(self, index):
        return '%d' % (index,)

    @property
    def _values(self):
        """Collect values for TrackableDataStructure."""
        return self

    def append(self, value):
        """Add a new trackable value."""
        value = self._track_value(value, self._name_element(len(self._storage)))
        self._storage.append(value)

    def extend(self, values):
        """Add a sequence of trackable values."""
        for value in values:
            self.append(value)

    def __iadd__(self, values):
        self.extend(values)
        return self

    def __add__(self, other):
        return self._storage + getattr(other, '_storage', other)

    def __imul__(self, y):
        if y <= 0:
            raise ValueError(f'List only supports append, multiplying in place by {y} removes elements.')
        n = len(self._storage)
        for _ in range(y - 1):
            for i in range(n):
                self.append(self._storage[i])
        return self

    def __mul__(self, n):
        return self._storage * n

    def __rmul__(self, n):
        return self * n

    def __radd__(self, other):
        return other + self._storage

    def __getitem__(self, key):
        return self._storage[key]

    def __getslice__(self, i, j):
        return self._storage[slice(i, j)]

    def __len__(self):
        return len(self._storage)

    def __repr__(self):
        return 'List(%s)' % (repr(self._storage),)

    def __sizeof__(self):
        return super().__sizeof__() + sys.getsizeof(self._storage)