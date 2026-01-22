import abc
import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.trackable import base as trackable
class PreprocessingLayer(Layer, metaclass=abc.ABCMeta):
    """Base class for Preprocessing Layers.

  **Don't use this class directly: it's an abstract base class!** You may
  be looking for one of the many built-in
  [preprocessing layers](https://keras.io/guides/preprocessing_layers/)
  instead.

  Preprocessing layers are layers whose state gets computed before model
  training starts. They do not get updated during training.
  Most preprocessing layers implement an `adapt()` method for state computation.

  The `PreprocessingLayer` class is the base class you would subclass to
  implement your own preprocessing layers.

  Attributes:
    streaming: Whether a layer can be adapted multiple times without resetting
      the state of the layer.
  """
    _must_restore_from_config = True

    def __init__(self, streaming=True, **kwargs):
        super(PreprocessingLayer, self).__init__(**kwargs)
        self._streaming = streaming
        self._is_compiled = False
        self._is_adapted = False
        self._reset_state_impl = self.reset_state
        self.reset_state = self._reset_state_wrapper
        self._adapt_function = None

    @property
    def streaming(self):
        """Whether `adapt` can be called twice without resetting the state."""
        return self._streaming

    @property
    def is_adapted(self):
        """Whether the layer has been fit to data already."""
        return self._is_adapted

    def update_state(self, data):
        """Accumulates statistics for the preprocessing layer.

    Arguments:
      data: A mini-batch of inputs to the layer.
    """
        raise NotImplementedError

    def reset_state(self):
        """Resets the statistics of the preprocessing layer."""
        raise NotImplementedError

    def merge_state(self, layers):
        """Merge the statistics of multiple preprocessing layers.

    This layer will contain the merged state.

    Arguments:
      layers: Layers whose statistics should be merge with the statistics of
        this layer.
    """
        raise NotImplementedError

    def finalize_state(self):
        """Finalize the statistics for the preprocessing layer.

    This method is called at the end of `adapt` or after restoring a serialized
    preprocessing layer's state. This method handles any one-time operations
    that should occur on the layer's state before `Layer.__call__`.
    """
        pass

    def make_adapt_function(self):
        """Creates a function to execute one step of `adapt`.

    This method can be overridden to support custom adapt logic.
    This method is called by `PreprocessingLayer.adapt`.

    Typically, this method directly controls `tf.function` settings,
    and delegates the actual state update logic to
    `PreprocessingLayer.update_state`.

    This function is cached the first time `PreprocessingLayer.adapt`
    is called. The cache is cleared whenever `PreprocessingLayer.compile`
    is called.

    Returns:
      Function. The function created by this method should accept a
      `tf.data.Iterator`, retrieve a batch, and update the state of the
      layer.
    """
        if self._adapt_function is not None:
            return self._adapt_function

        def adapt_step(iterator):
            data = next(iterator)
            self._adapt_maybe_build(data)
            self.update_state(data)
        if self._steps_per_execution.numpy().item() == 1:
            adapt_fn = adapt_step
        else:

            def adapt_fn(iterator):
                for _ in math_ops.range(self._steps_per_execution):
                    adapt_step(iterator)
        if not self._run_eagerly:
            adapt_fn = def_function.function(adapt_fn)
        self._adapt_function = adapt_fn
        return self._adapt_function

    def compile(self, run_eagerly=None, steps_per_execution=None):
        """Configures the layer for `adapt`.

    Arguments:
      run_eagerly: Bool. Defaults to `False`. If `True`, this `Model`'s logic
        will not be wrapped in a `tf.function`. Recommended to leave this as
        `None` unless your `Model` cannot be run inside a `tf.function`.
        steps_per_execution: Int. Defaults to 1. The number of batches to run
          during each `tf.function` call. Running multiple batches inside a
          single `tf.function` call can greatly improve performance on TPUs or
          small models with a large Python overhead.
    """
        if steps_per_execution is None:
            steps_per_execution = 1
        self._configure_steps_per_execution(steps_per_execution)
        if run_eagerly is None:
            run_eagerly = self.dynamic
        self._run_eagerly = run_eagerly
        self._is_compiled = True

    def adapt(self, data, batch_size=None, steps=None, reset_state=True):
        """Fits the state of the preprocessing layer to the data being passed.

    After calling `adapt` on a layer, a preprocessing layer's state will not
    update during training. In order to make preprocessing layers efficient in
    any distribution context, they are kept constant with respect to any
    compiled `tf.Graph`s that call the layer. This does not affect the layer use
    when adapting each layer only once, but if you adapt a layer multiple times
    you will need to take care to re-compile any compiled functions as follows:

     * If you are adding a preprocessing layer to a `keras.Model`, you need to
       call `model.compile` after each subsequent call to `adapt`.
     * If you are calling a preprocessing layer inside `tf.data.Dataset.map`,
       you should call `map` again on the input `tf.data.Dataset` after each
       `adapt`.
     * If you are using a `tf.function` directly which calls a preprocessing
       layer, you need to call `tf.function` again on your callable after
       each subsequent call to `adapt`.

    `tf.keras.Model` example with multiple adapts:

    >>> layer = tf.keras.layers.experimental.preprocessing.Normalization(
    ...     axis=None)
    >>> layer.adapt([0, 2])
    >>> model = tf.keras.Sequential(layer)
    >>> model.predict([0, 1, 2])
    array([-1.,  0.,  1.], dtype=float32)
    >>> layer.adapt([-1, 1])
    >>> model.compile() # This is needed to re-compile model.predict!
    >>> model.predict([0, 1, 2])
    array([0., 1., 2.], dtype=float32)

    `tf.data.Dataset` example with multiple adapts:

    >>> layer = tf.keras.layers.experimental.preprocessing.Normalization(
    ...     axis=None)
    >>> layer.adapt([0, 2])
    >>> input_ds = tf.data.Dataset.range(3)
    >>> normalized_ds = input_ds.map(layer)
    >>> list(normalized_ds.as_numpy_iterator())
    [array([-1.], dtype=float32),
     array([0.], dtype=float32),
     array([1.], dtype=float32)]
    >>> layer.adapt([-1, 1])
    >>> normalized_ds = input_ds.map(layer) # Re-map over the input dataset.
    >>> list(normalized_ds.as_numpy_iterator())
    [array([0.], dtype=float32),
     array([1.], dtype=float32),
     array([2.], dtype=float32)]

    Arguments:
        data: The data to train on. It can be passed either as a tf.data
          Dataset, or as a numpy array.
        batch_size: Integer or `None`.
            Number of samples per state update.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of datasets, generators, or `keras.utils.Sequence` instances
            (since they generate batches).
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            When training with input tensors such as
            TensorFlow data tensors, the default `None` is equal to
            the number of samples in your dataset divided by
            the batch size, or 1 if that cannot be determined. If x is a
            `tf.data` dataset, and 'steps' is None, the epoch will run until
            the input dataset is exhausted. When passing an infinitely
            repeating dataset, you must specify the `steps` argument. This
            argument is not supported with array inputs.
        reset_state: Optional argument specifying whether to clear the state of
          the layer at the start of the call to `adapt`, or whether to start
          from the existing state. This argument may not be relevant to all
          preprocessing layers: a subclass of PreprocessingLayer may choose to
          throw if 'reset_state' is set to False.
    """
        _disallow_inside_tf_function('adapt')
        if not version_utils.should_use_v2():
            raise RuntimeError('`adapt` is only supported in tensorflow v2.')
        if not self.streaming and self._is_adapted and (not reset_state):
            raise ValueError('{} does not supporting calling `adapt` twice without resetting the state.'.format(self.__class__.__name__))
        if not self._is_compiled:
            self.compile()
        if self.built and reset_state:
            self.reset_state()
        data_handler = data_adapter.DataHandler(data, batch_size=batch_size, steps_per_epoch=steps, epochs=1, steps_per_execution=self._steps_per_execution, distribute=False)
        self._adapt_function = self.make_adapt_function()
        for _, iterator in data_handler.enumerate_epochs():
            with data_handler.catch_stop_iteration():
                for _ in data_handler.steps():
                    self._adapt_function(iterator)
                    if data_handler.should_sync:
                        context.async_wait()
        self.finalize_state()
        self._is_adapted = True

    def _reset_state_wrapper(self):
        """Calls `reset_state` and sets `adapted` to `False`."""
        self._reset_state_impl()
        self._is_adapted = False

    @trackable.no_automatic_dependency_tracking
    def _configure_steps_per_execution(self, steps_per_execution):
        self._steps_per_execution = variables.Variable(steps_per_execution, dtype='int64', aggregation=variables.VariableAggregationV2.ONLY_FIRST_REPLICA)

    def _adapt_maybe_build(self, data):
        if not self.built:
            try:
                data_shape = data.shape
                data_shape_nones = tuple([None] * len(data.shape))
            except AttributeError:
                data_shape = None
                data_shape_nones = None
            batch_input_shape = getattr(self, '_batch_input_shape', None)
            if batch_input_shape is None:
                self._batch_input_shape = data_shape_nones
            self.build(data_shape)
            self.built = True