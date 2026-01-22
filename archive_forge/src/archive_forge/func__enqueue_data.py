from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import random
import types as tp
import numpy as np
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator.inputs.queues import feeding_queue_runner as fqr
def _enqueue_data(data, capacity, shuffle=False, min_after_dequeue=None, num_threads=1, seed=None, name='enqueue_input', enqueue_size=1, num_epochs=None, pad_value=None):
    """Creates a queue filled from a numpy array or pandas `DataFrame`.

    Returns a queue filled with the rows of the given (`OrderedDict` of) array
    or `DataFrame`. In the case of a pandas `DataFrame`, the first enqueued
    `Tensor` corresponds to the index of the `DataFrame`. For (`OrderedDict` of)
    numpy arrays, the first enqueued `Tensor` contains the row number.

  Args:
    data: a numpy `ndarray`, `OrderedDict` of numpy arrays, or a generator
      yielding `dict`s of numpy arrays or pandas `DataFrame` that will be read
      into the queue.
    capacity: the capacity of the queue.
    shuffle: whether or not to shuffle the rows of the array.
    min_after_dequeue: minimum number of elements that can remain in the queue
      after a dequeue operation. Only used when `shuffle` is true. If not set,
      defaults to `capacity` / 4.
    num_threads: number of threads used for reading and enqueueing.
    seed: used to seed shuffling and reader starting points.
    name: a scope name identifying the data.
    enqueue_size: the number of rows to enqueue per step.
    num_epochs: limit enqueuing to a specified number of epochs, if provided.
    pad_value: default value for dynamic padding of data samples, if provided.

  Returns:
    A queue filled with the rows of the given (`OrderedDict` of) array or
      `DataFrame`.

  Raises:
    TypeError: `data` is not a Pandas `DataFrame`, an `OrderedDict` of numpy
      arrays, a numpy `ndarray`, or a generator producing these.
    NotImplementedError: padding and shuffling data at the same time.
    NotImplementedError: padding usage with non generator data type.
  """
    with ops.name_scope(name):
        if isinstance(data, np.ndarray):
            types = [tf.dtypes.int64, tf.dtypes.as_dtype(data.dtype)]
            queue_shapes = [(), data.shape[1:]]
            get_feed_fn = _ArrayFeedFn
        elif isinstance(data, collections.OrderedDict):
            types = [tf.dtypes.int64] + [tf.dtypes.as_dtype(col.dtype) for col in data.values()]
            queue_shapes = [()] + [col.shape[1:] for col in data.values()]
            get_feed_fn = _OrderedDictNumpyFeedFn
        elif isinstance(data, tp.FunctionType):
            x_first_el = six.next(data())
            x_first_keys = sorted(x_first_el.keys())
            x_first_values = [x_first_el[key] for key in x_first_keys]
            types = [tf.dtypes.as_dtype(col.dtype) for col in x_first_values]
            queue_shapes = [col.shape for col in x_first_values]
            get_feed_fn = _GeneratorFeedFn
        elif HAS_PANDAS and isinstance(data, pd.DataFrame):
            types = [tf.dtypes.as_dtype(dt) for dt in [data.index.dtype] + list(data.dtypes)]
            queue_shapes = [() for _ in types]
            get_feed_fn = _PandasFeedFn
        else:
            raise TypeError('data must be either a numpy array or pandas DataFrame if pandas is installed; got {}'.format(type(data).__name__))
        pad_data = pad_value is not None
        if pad_data and get_feed_fn is not _GeneratorFeedFn:
            raise NotImplementedError('padding is only available with generator usage')
        if shuffle and pad_data:
            raise NotImplementedError('padding and shuffling data at the same time is not implemented')
        if num_threads > 1 and num_epochs is not None:
            tf.compat.v1.logging.warn('enqueue_data was called with num_epochs and num_threads > 1. num_epochs is applied per thread, so this will produce more epochs than you probably intend. If you want to limit epochs, use one thread.')
        if shuffle and num_threads > 1 and (num_epochs is not None):
            tf.compat.v1.logging.warn('enqueue_data was called with shuffle=True, num_threads > 1, and num_epochs. This will create multiple threads, all reading the array/dataframe in order adding to the same shuffling queue; the results will likely not be sufficiently shuffled.')
        if not shuffle and num_threads > 1:
            tf.compat.v1.logging.warn('enqueue_data was called with shuffle=False and num_threads > 1. This will create multiple threads, all reading the array/dataframe in order. If you want examples read in order, use one thread; if you want multiple threads, enable shuffling.')
        if shuffle:
            min_after_dequeue = int(capacity / 4 if min_after_dequeue is None else min_after_dequeue)
            queue = tf.queue.RandomShuffleQueue(capacity, min_after_dequeue, dtypes=types, shapes=queue_shapes, seed=seed)
        elif pad_data:
            min_after_dequeue = 0
            queue_shapes = list(map(lambda x: tuple(list(x[:-1]) + [None]) if len(x) > 0 else x, queue_shapes))
            queue = tf.queue.PaddingFIFOQueue(capacity, dtypes=types, shapes=queue_shapes)
        else:
            min_after_dequeue = 0
            queue = tf.queue.FIFOQueue(capacity, dtypes=types, shapes=queue_shapes)
        enqueue_ops = []
        feed_fns = []
        for i in range(num_threads):
            placeholders = [tf.compat.v1.placeholder(t) for t in types]
            enqueue_ops.append(queue.enqueue_many(placeholders))
            seed_i = None if seed is None else (i + 1) * seed
            if not pad_data:
                feed_fns.append(get_feed_fn(placeholders, data, enqueue_size, random_start=shuffle, seed=seed_i, num_epochs=num_epochs))
            else:
                feed_fns.append(get_feed_fn(placeholders, data, enqueue_size, random_start=shuffle, seed=seed_i, num_epochs=num_epochs, pad_value=pad_value))
        runner = fqr._FeedingQueueRunner(queue=queue, enqueue_ops=enqueue_ops, feed_fns=feed_fns)
        tf.compat.v1.train.queue_runner.add_queue_runner(runner)
        full = tf.cast(tf.math.maximum(0, queue.size() - min_after_dequeue), tf.dtypes.float32) * (1.0 / (capacity - min_after_dequeue))
        summary_name = 'queue/%sfraction_over_%d_of_%d_full' % (queue.name, min_after_dequeue, capacity - min_after_dequeue)
        tf.compat.v1.summary.scalar(summary_name, full)
        return queue