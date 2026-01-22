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
@tf_export('queue.QueueBase', v1=['queue.QueueBase', 'io.QueueBase', 'QueueBase'])
@deprecation.deprecated_endpoints(['io.QueueBase', 'QueueBase'])
class QueueBase:
    """Base class for queue implementations.

  A queue is a TensorFlow data structure that stores tensors across
  multiple steps, and exposes operations that enqueue and dequeue
  tensors.

  Each queue element is a tuple of one or more tensors, where each
  tuple component has a static dtype, and may have a static shape. The
  queue implementations support versions of enqueue and dequeue that
  handle single elements, versions that support enqueuing and
  dequeuing a batch of elements at once.

  See `tf.queue.FIFOQueue` and
  `tf.queue.RandomShuffleQueue` for concrete
  implementations of this class, and instructions on how to create
  them.
  """

    def __init__(self, dtypes, shapes, names, queue_ref):
        """Constructs a queue object from a queue reference.

    The two optional lists, `shapes` and `names`, must be of the same length
    as `dtypes` if provided.  The values at a given index `i` indicate the
    shape and name to use for the corresponding queue component in `dtypes`.

    Args:
      dtypes:  A list of types.  The length of dtypes must equal the number
        of tensors in each element.
      shapes: Constraints on the shapes of tensors in an element:
        A list of shape tuples or None. This list is the same length
        as dtypes.  If the shape of any tensors in the element are constrained,
        all must be; shapes can be None if the shapes should not be constrained.
      names: Optional list of names.  If provided, the `enqueue()` and
        `dequeue()` methods will use dictionaries with these names as keys.
        Must be None or a list or tuple of the same length as `dtypes`.
      queue_ref: The queue reference, i.e. the output of the queue op.

    Raises:
      ValueError: If one of the arguments is invalid.
    """
        self._dtypes = dtypes
        if shapes is not None:
            if len(shapes) != len(dtypes):
                raise ValueError(f'Queue shapes must have the same length as dtypes, received len(shapes)={len(shapes)}, len(dtypes)={len(dtypes)}')
            self._shapes = [tensor_shape.TensorShape(s) for s in shapes]
        else:
            self._shapes = [tensor_shape.unknown_shape() for _ in self._dtypes]
        if names is not None:
            if len(names) != len(dtypes):
                raise ValueError(f'Queue names must have the same length as dtypes,received len(names)={len(names)},len {len(dtypes)}')
            self._names = names
        else:
            self._names = None
        self._queue_ref = queue_ref
        if isinstance(queue_ref, ops.EagerTensor):
            if context.context().scope_name:
                self._name = context.context().scope_name
            else:
                self._name = 'Empty'
            self._resource_deleter = resource_variable_ops.EagerResourceDeleter(queue_ref, None)
        else:
            self._name = self._queue_ref.op.name.split('/')[-1]

    @staticmethod
    def from_list(index, queues):
        """Create a queue using the queue reference from `queues[index]`.

    Args:
      index: An integer scalar tensor that determines the input that gets
        selected.
      queues: A list of `QueueBase` objects.

    Returns:
      A `QueueBase` object.

    Raises:
      TypeError: When `queues` is not a list of `QueueBase` objects,
        or when the data types of `queues` are not all the same.
    """
        if not queues or not isinstance(queues, list) or (not all((isinstance(x, QueueBase) for x in queues))):
            raise TypeError('A list of queues expected')
        dtypes = queues[0].dtypes
        if not all((dtypes == q.dtypes for q in queues[1:])):
            raise TypeError('Queues do not have matching component dtypes.')
        names = queues[0].names
        if not all((names == q.names for q in queues[1:])):
            raise TypeError('Queues do not have matching component names.')
        queue_shapes = [q.shapes for q in queues]
        reduced_shapes = [functools.reduce(_shape_common, s) for s in zip(*queue_shapes)]
        queue_refs = array_ops_stack.stack([x.queue_ref for x in queues])
        selected_queue = array_ops.gather(queue_refs, index)
        return QueueBase(dtypes=dtypes, shapes=reduced_shapes, names=names, queue_ref=selected_queue)

    @property
    def queue_ref(self):
        """The underlying queue reference."""
        return self._queue_ref

    @property
    def name(self):
        """The name of the underlying queue."""
        if context.executing_eagerly():
            return self._name
        return self._queue_ref.op.name

    @property
    def dtypes(self):
        """The list of dtypes for each component of a queue element."""
        return self._dtypes

    @property
    def shapes(self):
        """The list of shapes for each component of a queue element."""
        return self._shapes

    @property
    def names(self):
        """The list of names for each component of a queue element."""
        return self._names

    def _check_enqueue_dtypes(self, vals):
        """Validate and convert `vals` to a list of `Tensor`s.

    The `vals` argument can be a Tensor, a list or tuple of tensors, or a
    dictionary with tensor values.

    If it is a dictionary, the queue must have been constructed with a
    `names` attribute and the dictionary keys must match the queue names.
    If the queue was constructed with a `names` attribute, `vals` must
    be a dictionary.

    Args:
      vals: A tensor, a list or tuple of tensors, or a dictionary..

    Returns:
      A list of `Tensor` objects.

    Raises:
      ValueError: If `vals` is invalid.
    """
        if isinstance(vals, dict):
            if not self._names:
                raise ValueError('Queue must have names to enqueue a dictionary')
            if sorted(self._names, key=str) != sorted(vals.keys(), key=str):
                raise ValueError(f'Keys in dictionary to enqueue do not match names of Queue.  Dictionary: {sorted(vals.keys())},Queue: {sorted(self._names)}')
            vals = [vals[k] for k in self._names]
        else:
            if self._names:
                raise ValueError('You must enqueue a dictionary in a Queue with names')
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
        tensors = []
        for i, (val, dtype) in enumerate(zip(vals, self._dtypes)):
            tensors.append(ops.convert_to_tensor(val, dtype=dtype, name='component_%d' % i))
        return tensors

    def _scope_vals(self, vals):
        """Return a list of values to pass to `name_scope()`.

    Args:
      vals: A tensor, a list or tuple of tensors, or a dictionary.

    Returns:
      The values in vals as a list.
    """
        if isinstance(vals, (list, tuple)):
            return vals
        elif isinstance(vals, dict):
            return vals.values()
        else:
            return [vals]

    def enqueue(self, vals, name=None):
        """Enqueues one element to this queue.

    If the queue is full when this operation executes, it will block
    until the element has been enqueued.

    At runtime, this operation may raise an error if the queue is
    `tf.QueueBase.close` before or during its execution. If the
    queue is closed before this operation runs,
    `tf.errors.CancelledError` will be raised. If this operation is
    blocked, and either (i) the queue is closed by a close operation
    with `cancel_pending_enqueues=True`, or (ii) the session is
    `tf.Session.close`,
    `tf.errors.CancelledError` will be raised.

    Args:
      vals: A tensor, a list or tuple of tensors, or a dictionary containing
        the values to enqueue.
      name: A name for the operation (optional).

    Returns:
      The operation that enqueues a new tuple of tensors to the queue.
    """
        with ops.name_scope(name, '%s_enqueue' % self._name, self._scope_vals(vals)) as scope:
            vals = self._check_enqueue_dtypes(vals)
            for val, shape in zip(vals, self._shapes):
                val.get_shape().assert_is_compatible_with(shape)
            if self._queue_ref.dtype == _dtypes.resource:
                return gen_data_flow_ops.queue_enqueue_v2(self._queue_ref, vals, name=scope)
            else:
                return gen_data_flow_ops.queue_enqueue(self._queue_ref, vals, name=scope)

    def enqueue_many(self, vals, name=None):
        """Enqueues zero or more elements to this queue.

    This operation slices each component tensor along the 0th dimension to
    make multiple queue elements. All of the tensors in `vals` must have the
    same size in the 0th dimension.

    If the queue is full when this operation executes, it will block
    until all of the elements have been enqueued.

    At runtime, this operation may raise an error if the queue is
    `tf.QueueBase.close` before or during its execution. If the
    queue is closed before this operation runs,
    `tf.errors.CancelledError` will be raised. If this operation is
    blocked, and either (i) the queue is closed by a close operation
    with `cancel_pending_enqueues=True`, or (ii) the session is
    `tf.Session.close`,
    `tf.errors.CancelledError` will be raised.

    Args:
      vals: A tensor, a list or tuple of tensors, or a dictionary
        from which the queue elements are taken.
      name: A name for the operation (optional).

    Returns:
      The operation that enqueues a batch of tuples of tensors to the queue.
    """
        with ops.name_scope(name, '%s_EnqueueMany' % self._name, self._scope_vals(vals)) as scope:
            vals = self._check_enqueue_dtypes(vals)
            batch_dim = tensor_shape.dimension_value(vals[0].get_shape().with_rank_at_least(1)[0])
            batch_dim = tensor_shape.Dimension(batch_dim)
            for val, shape in zip(vals, self._shapes):
                val_batch_dim = tensor_shape.dimension_value(val.get_shape().with_rank_at_least(1)[0])
                val_batch_dim = tensor_shape.Dimension(val_batch_dim)
                batch_dim = batch_dim.merge_with(val_batch_dim)
                val.get_shape()[1:].assert_is_compatible_with(shape)
            return gen_data_flow_ops.queue_enqueue_many_v2(self._queue_ref, vals, name=scope)

    def _dequeue_return_value(self, tensors):
        """Return the value to return from a dequeue op.

    If the queue has names, return a dictionary with the
    names as keys.  Otherwise return either a single tensor
    or a list of tensors depending on the length of `tensors`.

    Args:
      tensors: List of tensors from the dequeue op.

    Returns:
      A single tensor, a list of tensors, or a dictionary
      of tensors.
    """
        if self._names:
            return {n: tensors[i] for i, n in enumerate(self._names)}
        elif len(tensors) == 1:
            return tensors[0]
        else:
            return tensors

    def dequeue(self, name=None):
        """Dequeues one element from this queue.

    If the queue is empty when this operation executes, it will block
    until there is an element to dequeue.

    At runtime, this operation may raise an error if the queue is
    `tf.QueueBase.close` before or during its execution. If the
    queue is closed, the queue is empty, and there are no pending
    enqueue operations that can fulfill this request,
    `tf.errors.OutOfRangeError` will be raised. If the session is
    `tf.Session.close`,
    `tf.errors.CancelledError` will be raised.

    Args:
      name: A name for the operation (optional).

    Returns:
      The tuple of tensors that was dequeued.
    """
        if name is None:
            name = '%s_Dequeue' % self._name
        if self._queue_ref.dtype == _dtypes.resource:
            ret = gen_data_flow_ops.queue_dequeue_v2(self._queue_ref, self._dtypes, name=name)
        else:
            ret = gen_data_flow_ops.queue_dequeue(self._queue_ref, self._dtypes, name=name)
        if not context.executing_eagerly():
            op = ret[0].op
            for output, shape in zip(op.values(), self._shapes):
                output.set_shape(shape)
        return self._dequeue_return_value(ret)

    def dequeue_many(self, n, name=None):
        """Dequeues and concatenates `n` elements from this queue.

    This operation concatenates queue-element component tensors along
    the 0th dimension to make a single component tensor.  All of the
    components in the dequeued tuple will have size `n` in the 0th dimension.

    If the queue is closed and there are less than `n` elements left, then an
    `OutOfRange` exception is raised.

    At runtime, this operation may raise an error if the queue is
    `tf.QueueBase.close` before or during its execution. If the
    queue is closed, the queue contains fewer than `n` elements, and
    there are no pending enqueue operations that can fulfill this
    request, `tf.errors.OutOfRangeError` will be raised. If the
    session is `tf.Session.close`,
    `tf.errors.CancelledError` will be raised.

    Args:
      n: A scalar `Tensor` containing the number of elements to dequeue.
      name: A name for the operation (optional).

    Returns:
      The list of concatenated tensors that was dequeued.
    """
        if name is None:
            name = '%s_DequeueMany' % self._name
        ret = gen_data_flow_ops.queue_dequeue_many_v2(self._queue_ref, n=n, component_types=self._dtypes, name=name)
        if not context.executing_eagerly():
            op = ret[0].op
            batch_dim = tensor_shape.Dimension(tensor_util.constant_value(op.inputs[1]))
            for output, shape in zip(op.values(), self._shapes):
                output.set_shape(tensor_shape.TensorShape([batch_dim]).concatenate(shape))
        return self._dequeue_return_value(ret)

    def dequeue_up_to(self, n, name=None):
        """Dequeues and concatenates `n` elements from this queue.

    **Note** This operation is not supported by all queues.  If a queue does not
    support DequeueUpTo, then a `tf.errors.UnimplementedError` is raised.

    This operation concatenates queue-element component tensors along
    the 0th dimension to make a single component tensor. If the queue
    has not been closed, all of the components in the dequeued tuple
    will have size `n` in the 0th dimension.

    If the queue is closed and there are more than `0` but fewer than
    `n` elements remaining, then instead of raising a
    `tf.errors.OutOfRangeError` like `tf.QueueBase.dequeue_many`,
    less than `n` elements are returned immediately.  If the queue is
    closed and there are `0` elements left in the queue, then a
    `tf.errors.OutOfRangeError` is raised just like in `dequeue_many`.
    Otherwise the behavior is identical to `dequeue_many`.

    Args:
      n: A scalar `Tensor` containing the number of elements to dequeue.
      name: A name for the operation (optional).

    Returns:
      The tuple of concatenated tensors that was dequeued.
    """
        if name is None:
            name = '%s_DequeueUpTo' % self._name
        ret = gen_data_flow_ops.queue_dequeue_up_to_v2(self._queue_ref, n=n, component_types=self._dtypes, name=name)
        if not context.executing_eagerly():
            op = ret[0].op
            for output, shape in zip(op.values(), self._shapes):
                output.set_shape(tensor_shape.TensorShape([None]).concatenate(shape))
        return self._dequeue_return_value(ret)

    def close(self, cancel_pending_enqueues=False, name=None):
        """Closes this queue.

    This operation signals that no more elements will be enqueued in
    the given queue. Subsequent `enqueue` and `enqueue_many`
    operations will fail. Subsequent `dequeue` and `dequeue_many`
    operations will continue to succeed if sufficient elements remain
    in the queue. Subsequently dequeue and dequeue_many operations
    that would otherwise block waiting for more elements (if close
    hadn't been called) will now fail immediately.

    If `cancel_pending_enqueues` is `True`, all pending requests will also
    be canceled.

    Args:
      cancel_pending_enqueues: (Optional.) A boolean, defaulting to
        `False` (described above).
      name: A name for the operation (optional).

    Returns:
      The operation that closes the queue.
    """
        if name is None:
            name = '%s_Close' % self._name
        if self._queue_ref.dtype == _dtypes.resource:
            return gen_data_flow_ops.queue_close_v2(self._queue_ref, cancel_pending_enqueues=cancel_pending_enqueues, name=name)
        else:
            return gen_data_flow_ops.queue_close(self._queue_ref, cancel_pending_enqueues=cancel_pending_enqueues, name=name)

    def is_closed(self, name=None):
        """Returns true if queue is closed.

    This operation returns true if the queue is closed and false if the queue
    is open.

    Args:
      name: A name for the operation (optional).

    Returns:
      True if the queue is closed and false if the queue is open.
    """
        if name is None:
            name = '%s_Is_Closed' % self._name
        if self._queue_ref.dtype == _dtypes.resource:
            return gen_data_flow_ops.queue_is_closed_v2(self._queue_ref, name=name)
        else:
            return gen_data_flow_ops.queue_is_closed_(self._queue_ref, name=name)

    def size(self, name=None):
        """Compute the number of elements in this queue.

    Args:
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this queue.
    """
        if name is None:
            name = '%s_Size' % self._name
        if self._queue_ref.dtype == _dtypes.resource:
            return gen_data_flow_ops.queue_size_v2(self._queue_ref, name=name)
        else:
            return gen_data_flow_ops.queue_size(self._queue_ref, name=name)