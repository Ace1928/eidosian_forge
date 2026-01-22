import functools
import hashlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.gen_data_flow_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class StagingArea(BaseStagingArea):
    """Class for staging inputs. No ordering guarantees.

  A `StagingArea` is a TensorFlow data structure that stores tensors across
  multiple steps, and exposes operations that can put and get tensors.

  Each `StagingArea` element is a tuple of one or more tensors, where each
  tuple component has a static dtype, and may have a static shape.

  The capacity of a `StagingArea` may be bounded or unbounded.
  It supports multiple concurrent producers and consumers; and
  provides exactly-once delivery.

  Each element of a `StagingArea` is a fixed-length tuple of tensors whose
  dtypes are described by `dtypes`, and whose shapes are optionally described
  by the `shapes` argument.

  If the `shapes` argument is specified, each component of a staging area
  element must have the respective fixed shape. If it is
  unspecified, different elements may have different shapes,

  It can be configured with a capacity in which case
  put(values) will block until space becomes available.

  Similarly, it can be configured with a memory limit which
  will block put(values) until space is available.
  This is mostly useful for limiting the number of tensors on
  devices such as GPUs.

  All get() and peek() commands block if the requested data
  is not present in the Staging Area.

  """

    def __init__(self, dtypes, shapes=None, names=None, shared_name=None, capacity=0, memory_limit=0):
        """Constructs a staging area object.

    The two optional lists, `shapes` and `names`, must be of the same length
    as `dtypes` if provided.  The values at a given index `i` indicate the
    shape and name to use for the corresponding queue component in `dtypes`.

    The device scope at the time of object creation determines where the
    storage for the `StagingArea` will reside.  Calls to `put` will incur a copy
    to this memory space, if necessary.  Tensors returned by `get` will be
    placed according to the device scope when `get` is called.

    Args:
      dtypes:  A list of types.  The length of dtypes must equal the number
        of tensors in each element.
      shapes: (Optional.) Constraints on the shapes of tensors in an element.
        A list of shape tuples or None. This list is the same length
        as dtypes.  If the shape of any tensors in the element are constrained,
        all must be; shapes can be None if the shapes should not be constrained.
      names: (Optional.) If provided, the `get()` and
        `put()` methods will use dictionaries with these names as keys.
        Must be None or a list or tuple of the same length as `dtypes`.
      shared_name: (Optional.) A name to be used for the shared object. By
        passing the same name to two different python objects they will share
        the underlying staging area. Must be a string.
      capacity: (Optional.) Maximum number of elements.
        An integer. If zero, the Staging Area is unbounded
      memory_limit: (Optional.) Maximum number of bytes of all tensors
        in the Staging Area.
        An integer. If zero, the Staging Area is unbounded

    Raises:
      ValueError: If one of the arguments is invalid.
    """
        super(StagingArea, self).__init__(dtypes, shapes, names, shared_name, capacity, memory_limit)

    def put(self, values, name=None):
        """Create an op that places a value into the staging area.

    This operation will block if the `StagingArea` has reached
    its capacity.

    Args:
      values: A single tensor, a list or tuple of tensors, or a dictionary with
        tensor values. The number of elements must match the length of the
        list provided to the dtypes argument when creating the StagingArea.
      name: A name for the operation (optional).

    Returns:
        The created op.

    Raises:
      ValueError: If the number or type of inputs don't match the staging area.
    """
        with ops.name_scope(name, '%s_put' % self._name, self._scope_vals(values)) as scope:
            if not isinstance(values, (list, tuple, dict)):
                values = [values]
            indices = list(range(len(values)))
            vals, _ = self._check_put_dtypes(values, indices)
            with ops.colocate_with(self._coloc_op):
                op = gen_data_flow_ops.stage(values=vals, shared_name=self._name, name=scope, capacity=self._capacity, memory_limit=self._memory_limit)
            return op

    def __internal_get(self, get_fn, name):
        with ops.colocate_with(self._coloc_op):
            ret = get_fn()
        indices = list(range(len(self._dtypes)))
        return self._get_return_value(ret, indices)

    def get(self, name=None):
        """Gets one element from this staging area.

    If the staging area is empty when this operation executes, it will block
    until there is an element to dequeue.

    Note that unlike others ops that can block, like the queue Dequeue
    operations, this can stop other work from happening.  To avoid this, the
    intended use is for this to be called only when there will be an element
    already available.  One method for doing this in a training loop would be to
    run a `put()` call during a warmup session.run call, and then call both
    `get()` and `put()` in each subsequent step.

    The placement of the returned tensor will be determined by the current
    device scope when this function is called.

    Args:
      name: A name for the operation (optional).

    Returns:
      The tuple of tensors that was gotten.
    """
        if name is None:
            name = '%s_get' % self._name
        fn = lambda: gen_data_flow_ops.unstage(dtypes=self._dtypes, shared_name=self._name, name=name, capacity=self._capacity, memory_limit=self._memory_limit)
        return self.__internal_get(fn, name)

    def peek(self, index, name=None):
        """Peeks at an element in the staging area.

    If the staging area is too small to contain the element at
    the specified index, it will block until enough elements
    are inserted to complete the operation.

    The placement of the returned tensor will be determined by
    the current device scope when this function is called.

    Args:
      index: The index of the tensor within the staging area
              to look up.
      name: A name for the operation (optional).

    Returns:
      The tuple of tensors that was gotten.
    """
        if name is None:
            name = '%s_peek' % self._name
        fn = lambda: gen_data_flow_ops.stage_peek(index, dtypes=self._dtypes, shared_name=self._name, name=name, capacity=self._capacity, memory_limit=self._memory_limit)
        return self.__internal_get(fn, name)

    def size(self, name=None):
        """Returns the number of elements in the staging area.

    Args:
        name: A name for the operation (optional)

    Returns:
        The created op
    """
        if name is None:
            name = '%s_size' % self._name
        return gen_data_flow_ops.stage_size(name=name, shared_name=self._name, dtypes=self._dtypes, capacity=self._capacity, memory_limit=self._memory_limit)

    def clear(self, name=None):
        """Clears the staging area.

    Args:
        name: A name for the operation (optional)

    Returns:
        The created op
    """
        if name is None:
            name = '%s_clear' % self._name
        return gen_data_flow_ops.stage_clear(name=name, shared_name=self._name, dtypes=self._dtypes, capacity=self._capacity, memory_limit=self._memory_limit)