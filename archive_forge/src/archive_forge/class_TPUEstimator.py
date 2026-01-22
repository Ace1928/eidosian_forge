from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import enum
import math
import os
import signal
import sys
import threading
import time
import tensorflow as tf
import numpy as np
import six
from six.moves import queue as Queue  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.core.framework import variable_pb2
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf.tpu import compilation_result_pb2 as tpu_compilation_result
from tensorflow.python.data.util import nest as data_nest
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import functional as tpu_functional
from tensorflow.python.tpu import preempted_hook
from tensorflow.python.tpu import session_support
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_gradient
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import evaluation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_inspect
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output as export_output_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import error_handling
from tensorflow_estimator.python.estimator.tpu import iteration_count_estimator
from tensorflow_estimator.python.estimator.tpu import tpu_config
from tensorflow_estimator.python.estimator.tpu import tpu_context
from tensorflow_estimator.python.estimator.tpu import util as util_lib
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import AdagradParameters  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import AdamParameters  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import EmbeddingConfigSpec  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import StochasticGradientDescentParameters  # pylint: disable=unused-import
@estimator_export(v1=['estimator.tpu.TPUEstimator'])
class TPUEstimator(estimator_lib.Estimator):
    """Estimator with TPU support.

  TPUEstimator also supports training on CPU and GPU. You don't need to define
  a separate `tf.estimator.Estimator`.

  TPUEstimator handles many of the details of running on TPU devices, such as
  replicating inputs and models for each core, and returning to host
  periodically to run hooks.

  TPUEstimator transforms a global batch size in params to a per-shard batch
  size when calling the `input_fn` and `model_fn`. Users should specify
  global batch size in constructor, and then get the batch size for each shard
  in `input_fn` and `model_fn` by `params['batch_size']`.

  - For training, `model_fn` gets per-core batch size; `input_fn` may get
    per-core or per-host batch size depending on `per_host_input_for_training`
    in `TPUConfig` (See docstring for TPUConfig for details).

  - For evaluation and prediction, `model_fn` gets per-core batch size and
    `input_fn` get per-host batch size.

  Evaluation
  ==========

  `model_fn` should return `TPUEstimatorSpec`, which expects the `eval_metrics`
  for TPU evaluation. If eval_on_tpu is False, the evaluation will execute on
  CPU or GPU; in this case the following discussion on TPU evaluation does not
  apply.

  `TPUEstimatorSpec.eval_metrics` is a tuple of `metric_fn` and `tensors`, where
  `tensors` could be a list of any nested structure of `Tensor`s (See
  `TPUEstimatorSpec` for details).  `metric_fn` takes the `tensors` and returns
  a dict from metric string name to the result of calling a metric function,
  namely a `(metric_tensor, update_op)` tuple.

  One can set `use_tpu` to `False` for testing. All training, evaluation, and
  predict will be executed on CPU. `input_fn` and `model_fn` will receive
  `train_batch_size` or `eval_batch_size` unmodified as `params['batch_size']`.

  Current limitations:
  --------------------

  1. TPU evaluation only works on a single host (one TPU worker) except
     BROADCAST mode.

  2. `input_fn` for evaluation should **NOT** raise an end-of-input exception
     (`OutOfRangeError` or `StopIteration`). And all evaluation steps and all
     batches should have the same size.

  Example (MNIST):
  ----------------

  ```
  # The metric Fn which runs on CPU.
  def metric_fn(labels, logits):
    predictions = tf.argmax(logits, 1)
    return {
      'accuracy': tf.compat.v1.metrics.precision(
          labels=labels, predictions=predictions),
    }

  # Your model Fn which runs on TPU (eval_metrics is list in this example)
  def model_fn(features, labels, mode, config, params):
    ...
    logits = ...

    if mode = tf.estimator.ModeKeys.EVAL:
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(metric_fn, [labels, logits]))

  # or specify the eval_metrics tensors as dict.
  def model_fn(features, labels, mode, config, params):
    ...
    final_layer_output = ...

    if mode = tf.estimator.ModeKeys.EVAL:
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(metric_fn, {
              'labels': labels,
              'logits': final_layer_output,
          }))
  ```

  Prediction
  ==========

  Prediction on TPU is an experimental feature to support large batch inference.
  It is not designed for latency-critical system. In addition, due to some
  usability issues, for prediction with small dataset, CPU `.predict`, i.e.,
  creating a new `TPUEstimator` instance with `use_tpu=False`, might be more
  convenient.

  Note: In contrast to TPU training/evaluation, the `input_fn` for prediction
  *should* raise an end-of-input exception (`OutOfRangeError` or
  `StopIteration`), which serves as the stopping signal to `TPUEstimator`. To be
  precise, the ops created by `input_fn` produce one batch of the data.
  The `predict()` API processes one batch at a time. When reaching the end of
  the data source, an end-of-input exception should be raised by one of these
  operations. The user usually does not need to do this manually. As long as the
  dataset is not repeated forever, the `tf.data` API will raise an end-of-input
  exception automatically after the last batch has been produced.

  Note: Estimator.predict returns a Python generator. Please consume all the
  data from the generator so that TPUEstimator can shutdown the TPU system
  properly for user.

  Current limitations:
  --------------------
  1. TPU prediction only works on a single host (one TPU worker).

  2. `input_fn` must return a `Dataset` instance rather than `features`. In
  fact, .train() and .evaluate() also support Dataset as return value.

  Example (MNIST):
  ----------------
  ```
  height = 32
  width = 32
  total_examples = 100

  def predict_input_fn(params):
    batch_size = params['batch_size']

    images = tf.random.uniform(
        [total_examples, height, width, 3], minval=-1, maxval=1)

    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.map(lambda images: {'image': images})

    dataset = dataset.batch(batch_size)
    return dataset

  def model_fn(features, labels, params, mode):
     # Generate predictions, called 'output', from features['image']

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
              'predictions': output,
              'is_padding': features['is_padding']
          })

  tpu_est = TPUEstimator(
      model_fn=model_fn,
      ...,
      predict_batch_size=16)

  # Fully consume the generator so that TPUEstimator can shutdown the TPU
  # system.
  for item in tpu_est.predict(input_fn=input_fn):
    # Filter out item if the `is_padding` is 1.
    # Process the 'predictions'
  ```

  Exporting
  =========

  `export_saved_model` exports 2 metagraphs, one with `saved_model.SERVING`, and
  another with `saved_model.SERVING` and `saved_model.TPU` tags. At serving
  time, these tags are used to select the appropriate metagraph to load.

  Before running the graph on TPU, the TPU system needs to be initialized. If
  TensorFlow Serving model-server is used, this is done automatically. If not,
  please use `session.run(tpu.initialize_system())`.

  There are two versions of the API: 1 or 2.

  In V1, the exported CPU graph is `model_fn` as it is. The exported TPU graph
  wraps `tpu.rewrite()` and `TPUPartitionedCallOp` around `model_fn` so
  `model_fn` is on TPU by default. To place ops on CPU,
  `tpu_replication.outside_compilation(host_call, logits)` can be used.

  Example:
  ----------------

  ```
  def model_fn(features, labels, mode, config, params):
    ...
    logits = ...
    export_outputs = {
      'logits': export_output_lib.PredictOutput(
        {'logits': logits})
    }

    def host_call(logits):
      class_ids = math_ops.argmax(logits)
      classes = string_ops.as_string(class_ids)
      export_outputs['classes'] =
        export_output_lib.ClassificationOutput(classes=classes)

    tpu_replication.outside_compilation(host_call, logits)

    ...
  ```

  In V2, `export_saved_model()` sets up `params['use_tpu']` flag to let the user
  know if the code is exporting to TPU (or not). When `params['use_tpu']` is
  `True`, users need to call `tpu.rewrite()`, `TPUPartitionedCallOp` and/or
  `batch_function()`.

  TIP: V2 is recommended as it is more flexible (eg: batching, etc).

  @compatibility(TF2)
  TPU Estimator manages its own TensorFlow graph and session, so it is not
  compatible with TF2 behaviors. We recommend that you migrate to the newer
  `tf.distribute.TPUStrategy`. See the
  [TPU guide](https://www.tensorflow.org/guide/tpu) for details.
  @end_compatibility
  """

    def __init__(self, model_fn=None, model_dir=None, config=None, params=None, use_tpu=True, train_batch_size=None, eval_batch_size=None, predict_batch_size=None, batch_axis=None, eval_on_tpu=True, export_to_tpu=True, export_to_cpu=True, warm_start_from=None, embedding_config_spec=None, export_saved_model_api_version=ExportSavedModelApiVersion.V1):
        """Constructs an `TPUEstimator` instance.

    Args:
      model_fn: Model function as required by `Estimator` which returns
        EstimatorSpec or TPUEstimatorSpec. `training_hooks`, 'evaluation_hooks',
        and `prediction_hooks` must not capure any TPU Tensor inside the
        model_fn.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model. If `None`, the model_dir in
        `config` will be used if set. If both are set, they must be same. If
        both are `None`, a temporary directory will be used.
      config: An `tpu_config.RunConfig` configuration object. Cannot be `None`.
      params: An optional `dict` of hyper parameters that will be passed into
        `input_fn` and `model_fn`.  Keys are names of parameters, values are
        basic python types. There are reserved keys for `TPUEstimator`,
        including 'batch_size'.
      use_tpu: A bool indicating whether TPU support is enabled. Currently, -
        TPU training and evaluation respect this bit, but eval_on_tpu can
        override execution of eval. See below.
      train_batch_size: An int representing the global training batch size.
        TPUEstimator transforms this global batch size to a per-shard batch
        size, as params['batch_size'], when calling `input_fn` and `model_fn`.
        Cannot be `None` if `use_tpu` is `True`. Must be divisible by total
        number of replicas.
      eval_batch_size: An int representing evaluation batch size. Must be
        divisible by total number of replicas.
      predict_batch_size: An int representing the prediction batch size. Must be
        divisible by total number of replicas.
      batch_axis: A python tuple of int values describing how each tensor
        produced by the Estimator `input_fn` should be split across the TPU
        compute shards. For example, if your input_fn produced (images, labels)
        where the images tensor is in `HWCN` format, your shard dimensions would
        be [3, 0], where 3 corresponds to the `N` dimension of your images
        Tensor, and 0 corresponds to the dimension along which to split the
        labels to match up with the corresponding images. If None is supplied,
        and per_host_input_for_training is True, batches will be sharded based
        on the major dimension. If tpu_config.per_host_input_for_training is
        False or `PER_HOST_V2`, batch_axis is ignored.
      eval_on_tpu: If False, evaluation runs on CPU or GPU. In this case, the
        model_fn must return `EstimatorSpec` when called with `mode` as `EVAL`.
      export_to_tpu: If True, `export_saved_model()` exports a metagraph for
        serving on TPU. Note that unsupported export modes such as EVAL will be
        ignored. For those modes, only a CPU model will be exported. Currently,
        export_to_tpu only supports PREDICT.
      export_to_cpu: If True, `export_saved_model()` exports a metagraph for
        serving on CPU.
      warm_start_from: Optional string filepath to a checkpoint or SavedModel to
        warm-start from, or a `tf.estimator.WarmStartSettings` object to fully
        configure warm-starting.  If the string filepath is provided instead of
        a `WarmStartSettings`, then all variables are warm-started, and it is
        assumed that vocabularies and Tensor names are unchanged.
      embedding_config_spec: Optional EmbeddingConfigSpec instance to support
        using TPU embedding.
      export_saved_model_api_version: an integer: 1 or 2. 1 corresponds to V1,
        2 corresponds to V2. (Defaults to V1). With
        V1, `export_saved_model()` adds rewrite() and TPUPartitionedCallOp() for
        user; while in v2, user is expected to add rewrite(),
        TPUPartitionedCallOp() etc in their model_fn.

    Raises:
      ValueError: `params` has reserved keys already.
    """
        if config is None or not isinstance(config, tpu_config.RunConfig):
            raise ValueError('`config` must be provided with type `tpu_config.RunConfig`')
        if params is not None and any((k in params for k in _RESERVED_PARAMS_KEYS)):
            raise ValueError('{} are reserved keys but existed in params {}.'.format(_RESERVED_PARAMS_KEYS, params))
        if use_tpu:
            if train_batch_size is None:
                raise ValueError('`train_batch_size` cannot be `None`')
            util_lib.check_positive_integer(train_batch_size, 'train_batch_size')
            if config.tpu_config.per_host_input_for_training is tpu_config.InputPipelineConfig.PER_SHARD_V1 and config.tpu_config.num_cores_per_replica:
                raise ValueError('Model parallelism only supports per host input for training. Please adjust TPURunconfig.per_host_input_for_training.')
            if eval_batch_size is not None:
                util_lib.check_positive_integer(eval_batch_size, 'eval_batch_size')
            if predict_batch_size is not None:
                util_lib.check_positive_integer(predict_batch_size, 'predict_batch_size')
            if embedding_config_spec:
                if config.tpu_config.per_host_input_for_training not in (tpu_config.InputPipelineConfig.PER_HOST_V1, tpu_config.InputPipelineConfig.PER_HOST_V2):
                    raise ValueError('Only PER_HOST_V1 and PER_HOST_V2 is supported when using TPU Embedding; got {}.'.format(config.tpu_config.per_host_input_for_training))
                self._embedding_from_feature_columns = embedding_config_spec.feature_columns is not None
        if not (use_tpu and eval_on_tpu) and embedding_config_spec and (embedding_config_spec.partition_strategy == 'mod'):
            raise ValueError('Mod sharding of embedding tables not supported on CPU.')
        _tpu_estimator_gauge.get_cell().set(True)
        estimator_lib._verify_model_fn_args(model_fn, params)
        model_function = self._augment_model_fn(model_fn, batch_axis)
        self._log_every_n_steps = config.log_step_count_steps
        config = config.replace(log_step_count_steps=None)
        params = params or {}
        super(TPUEstimator, self).__init__(model_fn=model_function, model_dir=model_dir, config=config, params=params, warm_start_from=warm_start_from)
        self._iterations_per_training_loop = util_lib.parse_iterations_per_loop(self._config.tpu_config.iterations_per_loop)
        if self._iterations_per_training_loop.unit == 'seconds':
            self._log_every_n_secs = self._iterations_per_training_loop.value
            self._log_every_n_steps = None
        elif self._iterations_per_training_loop.unit == 'count':
            if self._log_every_n_steps is not None:
                self._log_every_n_steps = int(math.ceil(float(self._log_every_n_steps) / self._iterations_per_training_loop.value))
            self._log_every_n_secs = None
        else:
            assert False, 'Invalid TPUConfig `iterations_per_loop` value. Indicates a bug in `iterations_per_loop` parsing.'
        self._ctx = tpu_context._get_tpu_context(self._config, train_batch_size, eval_batch_size, predict_batch_size, use_tpu, eval_on_tpu, embedding_config_spec)
        self._export_to_cpu = export_to_cpu
        self._export_to_tpu = export_to_tpu
        if not (isinstance(export_saved_model_api_version, ExportSavedModelApiVersion) or export_saved_model_api_version == 1 or export_saved_model_api_version == 2):
            raise ValueError('export_saved_model_api_version should be 1 or 2; got {}.'.format(export_saved_model_api_version))
        self._export_saved_model_api_version = export_saved_model_api_version
        self._is_input_fn_invoked = None
        self._rendezvous = {}

    def _add_meta_graph_for_mode(self, builder, input_receiver_fn_map, checkpoint_path, save_variables=True, mode=model_fn_lib.ModeKeys.PREDICT, export_tags=None, check_variables=True, strip_default_attrs=True):
        if self._export_to_tpu and mode != model_fn_lib.ModeKeys.PREDICT:
            tf.compat.v1.logging.warn('TPUEstimator only handles mode PREDICT for exporting when `export_to_tpu` is `True`; Mode {} will be ignored for TPU.'.format(mode))
        if not self._export_to_cpu and (not self._export_to_tpu):
            raise ValueError('One of export_to_cpu and export_to_tpu must be true.')
        if self._export_to_cpu:
            super(TPUEstimator, self)._add_meta_graph_for_mode(builder, input_receiver_fn_map, checkpoint_path, save_variables, mode=mode, export_tags=export_tags, check_variables=check_variables, strip_default_attrs=strip_default_attrs)
        if self._export_to_tpu and mode == model_fn_lib.ModeKeys.PREDICT:
            input_receiver_fn_map = {_INFERENCE_ON_TPU_MODE: input_receiver_fn_map[mode]}
            export_tags = [tf.saved_model.SERVING, tf.saved_model.TPU]
            mode = _INFERENCE_ON_TPU_MODE
            if not self._export_to_cpu:
                check_variables = save_variables = True
            else:
                check_variables = save_variables = False
            super(TPUEstimator, self)._add_meta_graph_for_mode(builder, input_receiver_fn_map, checkpoint_path, save_variables=save_variables, mode=mode, export_tags=export_tags, check_variables=check_variables, strip_default_attrs=strip_default_attrs)

    def _call_model_fn(self, features, labels, mode, config):
        if mode == _INFERENCE_ON_TPU_MODE:
            context = tpu._TPUInferenceContext('tpu_inference', check_ops=False)
            try:
                context.Enter()
                if self._export_saved_model_api_version == ExportSavedModelApiVersion.V1 or self._export_saved_model_api_version == 1:
                    result = self._call_model_fn_for_inference(features, labels, mode, config)
                else:
                    result = super(TPUEstimator, self)._call_model_fn(features, labels, mode, config)
            finally:
                context.Exit()
            return result
        else:
            return super(TPUEstimator, self)._call_model_fn(features, labels, mode, config)

    def _call_model_fn_for_inference(self, features, labels, mode, config):
        """Wraps `_call_model_fn` for `export_saved_model`."""
        if mode != _INFERENCE_ON_TPU_MODE:
            raise ValueError('mode must be {}; got {}.'.format(_INFERENCE_ON_TPU_MODE, mode))
        return model_fn_inference_on_tpu(self._model_fn, features, labels, config, self._params, batch_config=None)

    def _create_global_step(self, graph):
        """Creates a global step suitable for TPUs.

    Args:
      graph: The graph in which to create the global step.

    Returns:
      A global step `Tensor`.

    Raises:
      ValueError: if the global step tensor is already defined.
    """
        return _create_global_step(graph)

    def _convert_train_steps_to_hooks(self, steps, max_steps):
        with self._ctx.with_mode(model_fn_lib.ModeKeys.TRAIN) as ctx:
            if ctx.is_running_on_cpu():
                return super(TPUEstimator, self)._convert_train_steps_to_hooks(steps, max_steps)
        if steps is None and max_steps is None:
            raise ValueError('For TPU training, one of `steps` or `max_steps` must be set. Cannot be both `None`.')
        if steps is not None:
            util_lib.check_positive_integer(steps, 'Train steps')
        if max_steps is not None:
            util_lib.check_positive_integer(max_steps, 'Train max_steps')
        return [_TPUStopAtStepHook(self._iterations_per_training_loop, steps, max_steps)]

    def _convert_eval_steps_to_hooks(self, steps):
        with self._ctx.with_mode(model_fn_lib.ModeKeys.EVAL) as ctx:
            if ctx.is_running_on_cpu():
                return super(TPUEstimator, self)._convert_eval_steps_to_hooks(steps)
        if steps is None:
            raise ValueError('Evaluate `steps` must be set on TPU. Cannot be `None`.')
        util_lib.check_positive_integer(steps, 'Eval steps')
        return [evaluation._StopAfterNEvalsHook(num_evals=steps), _SetEvalIterationsHook(steps)]

    def _call_input_fn(self, input_fn, mode, input_context=None):
        """Calls the input function.

    Args:
      input_fn: The input function.
      mode: ModeKeys
      input_context: Optional instance of `tf.distribute.InputContext`.

    Returns:
      In TPU mode, returns an input_fn to be called later in model_fn.
      Otherwise, calls the input_fn and returns either fatures or
        (features, labels).

    Raises:
      ValueError: if input_fn takes invalid arguments or does not have `params`.
    """
        input_fn_args = function_utils.fn_args(input_fn)
        config = self.config
        kwargs = {}
        if 'params' in input_fn_args:
            kwargs['params'] = self.params
        else:
            raise ValueError('input_fn ({}) does not include params argument, required by TPUEstimator to pass batch size as params["batch_size"]'.format(input_fn))
        if 'config' in input_fn_args:
            kwargs['config'] = config
        if 'mode' in input_fn_args:
            kwargs['mode'] = mode
        if 'input_context' in input_fn_args:
            kwargs['input_context'] = input_context
        self._is_input_fn_invoked = True
        with self._ctx.with_mode(mode) as ctx:
            if ctx.is_running_on_cpu() and ctx.is_input_slice_broadcast_to_all_cores():
                raise ValueError('Invalid TPUConfig `eval_training_input_configuration` value. SLICED mode only works on use_tpu = True.')
            batch_size_for_input_fn = ctx.batch_size_for_input_fn
            if batch_size_for_input_fn is not None:
                _add_item_to_params(kwargs['params'], _BATCH_SIZE_KEY, batch_size_for_input_fn)
            if ctx.is_running_on_cpu(is_export_mode=False):
                with tf.compat.v1.device('/device:CPU:0'):
                    return input_fn(**kwargs)

            def _input_fn(ctx):
                _add_item_to_params(kwargs['params'], _CTX_KEY, ctx)
                return input_fn(**kwargs)
            return _input_fn

    def _validate_features_in_predict_input(self, result):
        """Skip the validation.

    For TPUEstimator, we do not need to check the result type. `_InputPipeline`
    has stronger check. Parent class's check generates confusing warning msg.

    Args:
      result: `features` returned by input_fn.
    """
        pass

    def train(self, input_fn, hooks=None, steps=None, max_steps=None, saving_listeners=None):
        rendezvous = error_handling.ErrorRendezvous(num_sources=3)
        self._rendezvous[model_fn_lib.ModeKeys.TRAIN] = rendezvous
        try:
            return super(TPUEstimator, self).train(input_fn=input_fn, hooks=hooks, steps=steps, max_steps=max_steps, saving_listeners=saving_listeners)
        except Exception:
            rendezvous.record_error('training_loop', sys.exc_info())
        finally:
            rendezvous.record_done('training_loop')
            rendezvous.raise_errors()

    def evaluate(self, input_fn, steps=None, hooks=None, checkpoint_path=None, name=None):
        rendezvous = error_handling.ErrorRendezvous(num_sources=3)
        self._rendezvous[model_fn_lib.ModeKeys.EVAL] = rendezvous
        try:
            return super(TPUEstimator, self).evaluate(input_fn, steps=steps, hooks=hooks, checkpoint_path=checkpoint_path, name=name)
        except Exception:
            rendezvous.record_error('evaluation_loop', sys.exc_info())
        finally:
            rendezvous.record_done('evaluation_loop')
            rendezvous.raise_errors()

    def predict(self, input_fn, predict_keys=None, hooks=None, checkpoint_path=None, yield_single_examples=True):
        rendezvous = error_handling.ErrorRendezvous(num_sources=3)
        self._rendezvous[model_fn_lib.ModeKeys.PREDICT] = rendezvous
        try:
            for result in super(TPUEstimator, self).predict(input_fn=input_fn, predict_keys=predict_keys, hooks=hooks, checkpoint_path=checkpoint_path, yield_single_examples=yield_single_examples):
                yield result
        except Exception:
            rendezvous.record_error('prediction_loop', sys.exc_info())
        finally:
            rendezvous.record_done('prediction_loop')
            rendezvous.raise_errors()
        rendezvous.record_done('prediction_loop')
        rendezvous.raise_errors()

    def _augment_model_fn(self, model_fn, batch_axis):
        """Returns a new model_fn, which wraps the TPU support."""

        def _model_fn(features, labels, mode, config, params):
            """A Estimator `model_fn` for TPUEstimator."""
            if self._is_input_fn_invoked:
                is_export_mode = False
            else:
                is_export_mode = True
            self._is_input_fn_invoked = None
            if is_export_mode:
                if mode == _INFERENCE_ON_TPU_MODE:
                    _add_item_to_params(params, _USE_TPU_KEY, True)
                    mode = model_fn_lib.ModeKeys.PREDICT
                else:
                    _add_item_to_params(params, _USE_TPU_KEY, False)
            with self._ctx.with_mode(mode) as ctx:
                model_fn_wrapper = _ModelFnWrapper(model_fn, config, params, ctx)
                if self._log_every_n_steps is not None or self._log_every_n_secs is not None:
                    examples_hook = ExamplesPerSecondHook(ctx.global_batch_size, output_dir=self.model_dir if not config or config.save_summary_steps else None, every_n_steps=self._log_every_n_steps, every_n_secs=self._log_every_n_secs)
                if ctx.is_running_on_cpu(is_export_mode=is_export_mode):
                    tf.compat.v1.logging.info('Running %s on CPU/GPU', mode)
                    estimator_spec = model_fn_wrapper.call_without_tpu(features, labels, is_export_mode=is_export_mode)
                    if self._log_every_n_steps is not None or self._log_every_n_secs is not None:
                        estimator_spec = estimator_spec._replace(training_hooks=estimator_spec.training_hooks + (examples_hook,))
                    return estimator_spec
                assert labels is None, '`labels` passed to `model_fn` must be `None`.'
                assert callable(features), '`input_fn` is not callable.'
                input_fn = features
                tpu_init_ops = []
                if ctx.embedding_config and mode == model_fn_lib.ModeKeys.TRAIN:
                    dummy_table_variables, dummy_table_variables_init = tpu_embedding_gradient.create_dummy_table_variables(ctx.embedding_config.tpu_embedding)
                    ctx.embedding_config.dummy_table_variables = dummy_table_variables
                    tpu_init_ops.append(dummy_table_variables_init)
                input_holders = _InputPipeline(input_fn, batch_axis, ctx)
                enqueue_ops, dequeue_fn, input_hooks, run_infeed_loop_on_coordinator = input_holders.generate_infeed_enqueue_ops_and_dequeue_fn()
                graph = tf.compat.v1.get_default_graph()
                for enqueue_op in enqueue_ops:
                    if isinstance(enqueue_op, list):
                        graph.get_collection_ref(_TPU_ENQUEUE_OPS).extend(enqueue_op)
                    else:
                        graph.add_to_collection(_TPU_ENQUEUE_OPS, enqueue_op)
                if mode == model_fn_lib.ModeKeys.TRAIN:
                    compile_op, loss, host_call, scaffold_fn, training_hooks = _train_on_tpu_system(ctx, model_fn_wrapper, dequeue_fn)
                    has_saver_hook = training_hooks and any((isinstance(hook, tf.compat.v1.train.CheckpointSaverHook) for hook in training_hooks))
                    if ctx.embedding_config:
                        g = tf.compat.v1.get_default_graph()
                        table_to_config_dict = ctx.embedding_config.tpu_embedding.table_to_config_dict
                        optimization_parameters = ctx.embedding_config.tpu_embedding.optimization_parameters
                        if self._embedding_from_feature_columns:
                            embedding_variable_name_by_table, slot_variable_names_by_table = _tpu_estimator_embedding.get_full_variable_names(g, table_to_config_dict, optimization_parameters)
                        else:
                            embedding_variable_name_by_table = None
                            slot_variable_names_by_table = None
                        embedding_variables_and_ops = ctx.embedding_config.tpu_embedding.create_variables_and_ops(embedding_variable_name_by_table, slot_variable_names_by_table)
                        tpu_init_ops.extend(embedding_variables_and_ops.load_ops())
                    scaffold = _get_scaffold(scaffold_fn)
                    host_ops = host_call.create_tpu_hostcall()
                    shutdown_hooks = []
                    shutdown_mode = os.environ.get('TF_TPU_GRACEFUL_SHUTDOWN_MODE', 'reset_computation')
                    if shutdown_mode:
                        if shutdown_mode == 'shutdown_worker':
                            finalizer_hooks = [session_support.ShutdownLameWorkers()]
                        elif shutdown_mode == 'shutdown_all_workers':
                            finalizer_hooks = [session_support.ShutdownAllWorkers()]
                        elif shutdown_mode == 'reset_computation':
                            finalizer_hooks = [session_support.ResetComputation()]
                        elif not shutdown_mode:
                            finalizer_hooks = []
                        else:
                            raise ValueError('Unknown TF_TPU_GRACEFUL_SHUTDOWN_MODE "%s"' % shutdown_mode)
                        if finalizer_hooks:
                            if has_saver_hook:
                                saver = _NotSaver('No save on shutdown when there are user-defined CheckpointSaverHooks')
                            else:
                                saver = None
                            shutdown_hooks.append(session_support.GracefulShutdownHook(checkpoint_prefix=self.model_dir + '/model.ckpt', on_shutdown_hooks=finalizer_hooks, saver=saver))
                    with tf.control_dependencies([loss]):
                        global_step = tf.identity(tf.compat.v1.train.get_global_step())
                    hooks = input_hooks + shutdown_hooks
                    if ctx.feed_hook is not None:
                        tf.compat.v1.logging.info('Use user implemented tpu infeed outfeed session hook class.')
                        infeed_outfeed_session_hook_class = ctx.feed_hook
                    else:
                        infeed_outfeed_session_hook_class = TPUInfeedOutfeedSessionHook
                    hooks.extend([infeed_outfeed_session_hook_class(ctx, enqueue_ops, host_ops, tpu_compile_op=compile_op, run_infeed_loop_on_coordinator=run_infeed_loop_on_coordinator, rendezvous=self._rendezvous[mode], master=self._config.master, session_config=self._session_config, tpu_init_ops=tpu_init_ops, outfeed_every_n_steps=self._config.tpu_config.experimental_host_call_every_n_steps), InstallSignalHandlerHook()])
                    if _check_add_preemption_hook(self._config.cluster):
                        hooks.extend([preempted_hook.CloudTPUPreemptedHook(self._config.cluster)])
                    if self._log_every_n_steps is not None or self._log_every_n_secs is not None:
                        if self._iterations_per_training_loop.unit == 'count':
                            examples_hook._set_steps_per_run(self._iterations_per_training_loop.value)
                        hooks.append(tf.compat.v1.train.LoggingTensorHook({'loss': tf.identity(loss), 'step': global_step}, every_n_iter=self._log_every_n_steps, every_n_secs=self._log_every_n_secs))
                        hooks.append(examples_hook)
                    if training_hooks:
                        hooks.extend(training_hooks)
                    chief_hooks = []
                    if not has_saver_hook and (self._config.save_checkpoints_secs or self._config.save_checkpoints_steps):
                        checkpoint_hook = tf.compat.v1.train.CheckpointSaverHook(self.model_dir, save_secs=self._config.save_checkpoints_secs, save_steps=self._config.save_checkpoints_steps, scaffold=scaffold, save_graph_def=self._config.checkpoint_save_graph_def)
                        if self._iterations_per_training_loop.unit == 'count':
                            checkpoint_hook._set_steps_per_run(self._iterations_per_training_loop.value)
                        chief_hooks.append(checkpoint_hook)
                    else:
                        tf.compat.v1.logging.info('Bypassing TPUEstimator hook')
                    tf.compat.v1.summary.scalar(model_fn_lib.LOSS_METRIC_KEY, loss)
                    with tf.control_dependencies([loss]):
                        update_ops = _sync_variables_ops(ctx)
                        if ctx.embedding_config:
                            update_ops.extend(embedding_variables_and_ops.retrieve_ops())
                    _validate_tpu_training_graph(ctx)
                    train_op = tf.group(*update_ops)
                    graph.add_to_collection(_TPU_TRAIN_OP, train_op)
                    return model_fn_lib.EstimatorSpec(mode, loss=loss, training_chief_hooks=chief_hooks, training_hooks=hooks, train_op=train_op, scaffold=scaffold)
                if mode == model_fn_lib.ModeKeys.EVAL:
                    compile_op, total_loss, host_calls, scaffold_fn, eval_hooks = _eval_on_tpu_system(ctx, model_fn_wrapper, dequeue_fn)
                    if ctx.embedding_config:
                        g = tf.compat.v1.get_default_graph()
                        table_to_config_dict = ctx.embedding_config.tpu_embedding.table_to_config_dict
                        if self._embedding_from_feature_columns:
                            embedding_variable_name_by_table, _ = _tpu_estimator_embedding.get_full_variable_names(g, table_to_config_dict)
                        else:
                            embedding_variable_name_by_table = None
                        embedding_variables_and_ops = ctx.embedding_config.tpu_embedding.create_variables_and_ops(embedding_variable_name_by_table)
                        tpu_init_ops.extend(embedding_variables_and_ops.load_ops())
                    scaffold = _get_scaffold(scaffold_fn)
                    iterations_per_loop_var = _create_or_get_iterations_per_loop()
                    mean_loss = tf.compat.v1.div(total_loss, tf.cast(iterations_per_loop_var, dtype=total_loss.dtype))
                    with tf.control_dependencies([mean_loss]):
                        internal_ops_to_run = _sync_variables_ops(ctx)
                        internal_ops_to_run.append(_increase_eval_step_op(iterations_per_loop_var))
                    host_call_ret = host_calls.create_tpu_hostcall()
                    eval_metric_ops = {}
                    eval_update_ops = []
                    eval_metrics = host_call_ret.get('eval_metrics', {})
                    if eval_metrics:
                        with tf.control_dependencies(internal_ops_to_run):
                            dummy_update_op = tf.no_op()
                        for k, v in eval_metrics.items():
                            eval_metric_ops[k] = (v[0], dummy_update_op)
                            eval_update_ops.append(v[1])
                    else:
                        with tf.control_dependencies(internal_ops_to_run):
                            mean_loss = tf.identity(mean_loss)
                    if 'host_call' not in host_call_ret:
                        host_ops = []
                    else:
                        host_ops = host_call_ret['host_call']
                    hooks = [TPUInfeedOutfeedSessionHook(ctx, enqueue_ops, eval_update_ops + host_ops, tpu_compile_op=compile_op, run_infeed_loop_on_coordinator=run_infeed_loop_on_coordinator, rendezvous=self._rendezvous[mode], master=self._config.evaluation_master, session_config=self._session_config, tpu_init_ops=tpu_init_ops)] + input_hooks
                    if _check_add_preemption_hook(self._config.cluster):
                        hooks.extend([preempted_hook.CloudTPUPreemptedHook(self._config.cluster)])
                    if eval_hooks:
                        hooks.extend(eval_hooks)
                    return model_fn_lib.EstimatorSpec(mode, loss=mean_loss, evaluation_hooks=hooks, eval_metric_ops=eval_metric_ops, scaffold=scaffold)
                assert mode == model_fn_lib.ModeKeys.PREDICT
                compile_op, dummy_predict_op, host_calls, scaffold_fn, prediction_hooks = _predict_on_tpu_system(ctx, model_fn_wrapper, dequeue_fn)
                scaffold = _get_scaffold(scaffold_fn)
                with tf.control_dependencies([dummy_predict_op]):
                    internal_ops_to_run = _sync_variables_ops(ctx)
                    with tf.control_dependencies(internal_ops_to_run):
                        dummy_predict_op = tf.no_op()
                enqueue_ops.append(dummy_predict_op)
                host_call_ret = host_calls.create_tpu_hostcall()
                if 'host_call' not in host_call_ret:
                    host_ops = []
                else:
                    host_ops = host_call_ret['host_call']
                predictions = host_call_ret['predictions']
                _verify_cross_hosts_transfer_size(predictions, message='The estimated size for TPUEstimatorSpec.predictions is too large.')
                signals = host_call_ret['signals']
                with tf.control_dependencies(host_ops):
                    host_ops = []
                    scalar_stopping_signal = _StopSignals.as_scalar_stopping_signal(signals)
                    predictions = _PaddingSignals.slice_tensor_or_dict(predictions, signals)
                hooks = [_StoppingPredictHook(scalar_stopping_signal), TPUInfeedOutfeedSessionHookForPrediction(ctx, enqueue_ops, host_ops, rendezvous=self._rendezvous[mode], tpu_compile_op=compile_op, master=self._config.master, session_config=self._session_config)] + input_hooks
                if prediction_hooks:
                    hooks.extend(prediction_hooks)
                return model_fn_lib.EstimatorSpec(mode, prediction_hooks=hooks, predictions=predictions, scaffold=scaffold)
        return _model_fn