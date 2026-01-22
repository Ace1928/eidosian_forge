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
@tf_export('queue.PriorityQueue', v1=['queue.PriorityQueue', 'io.PriorityQueue', 'PriorityQueue'])
@deprecation.deprecated_endpoints(['io.PriorityQueue', 'PriorityQueue'])
class PriorityQueue(QueueBase):
    """A queue implementation that dequeues elements in prioritized order.

  See `tf.queue.QueueBase` for a description of the methods on
  this class.
  """

    def __init__(self, capacity, types, shapes=None, names=None, shared_name=None, name='priority_queue'):
        """Creates a queue that dequeues elements in a first-in first-out order.

    A `PriorityQueue` has bounded capacity; supports multiple concurrent
    producers and consumers; and provides exactly-once delivery.

    A `PriorityQueue` holds a list of up to `capacity` elements. Each
    element is a fixed-length tuple of tensors whose dtypes are
    described by `types`, and whose shapes are optionally described
    by the `shapes` argument.

    If the `shapes` argument is specified, each component of a queue
    element must have the respective fixed shape. If it is
    unspecified, different queue elements may have different shapes,
    but the use of `dequeue_many` is disallowed.

    Enqueues and Dequeues to the `PriorityQueue` must include an additional
    tuple entry at the beginning: the `priority`.  The priority must be
    an int64 scalar (for `enqueue`) or an int64 vector (for `enqueue_many`).

    Args:
      capacity: An integer. The upper bound on the number of elements
        that may be stored in this queue.
      types:  A list of `DType` objects. The length of `types` must equal
        the number of tensors in each queue element, except the first priority
        element.  The first tensor in each element is the priority,
        which must be type int64.
      shapes: (Optional.) A list of fully-defined `TensorShape` objects,
        with the same length as `types`, or `None`.
      names: (Optional.) A list of strings naming the components in the queue
        with the same length as `dtypes`, or `None`.  If specified, the dequeue
        methods return a dictionary with the names as keys.
      shared_name: (Optional.) If non-empty, this queue will be shared under
        the given name across multiple sessions.
      name: Optional name for the queue operation.
    """
        types = _as_type_list(types)
        shapes = _as_shape_list(shapes, types)
        queue_ref = gen_data_flow_ops.priority_queue_v2(component_types=types, shapes=shapes, capacity=capacity, shared_name=_shared_name(shared_name), name=name)
        priority_dtypes = [_dtypes.int64] + types
        priority_shapes = [()] + shapes if shapes else shapes
        super(PriorityQueue, self).__init__(priority_dtypes, priority_shapes, names, queue_ref)