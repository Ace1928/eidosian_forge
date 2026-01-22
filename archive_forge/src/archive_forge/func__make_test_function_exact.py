import copy
import itertools
import json
import warnings
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_ops
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import optimizers
from keras.src.dtensor import dtensor_api
from keras.src.dtensor import layout_map as layout_map_lib
from keras.src.engine import base_layer
from keras.src.engine import base_layer_utils
from keras.src.engine import compile_utils
from keras.src.engine import data_adapter
from keras.src.engine import input_layer as input_layer_module
from keras.src.engine import steps_per_execution_tuning
from keras.src.engine import training_utils
from keras.src.metrics import base_metric
from keras.src.mixed_precision import loss_scale_optimizer as lso
from keras.src.optimizers import optimizer
from keras.src.optimizers import optimizer_v1
from keras.src.saving import pickle_utils
from keras.src.saving import saving_api
from keras.src.saving import saving_lib
from keras.src.saving import serialization_lib
from keras.src.saving.legacy import serialization
from keras.src.saving.legacy.saved_model import json_utils
from keras.src.saving.legacy.saved_model import model_serialization
from keras.src.utils import generic_utils
from keras.src.utils import io_utils
from keras.src.utils import layer_utils
from keras.src.utils import tf_inspect
from keras.src.utils import tf_utils
from keras.src.utils import traceback_utils
from keras.src.utils import version_utils
from keras.src.utils.mode_keys import ModeKeys
def _make_test_function_exact(self):
    if getattr(self, '_shard_test_function', None):
        return self._shard_test_function

    def step_function(batch):

        def run_step(data):
            x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
            y_pred = self(x, training=False)
            return (x, y, y_pred, sample_weight)
        if self._jit_compile:
            run_step = tf.function(run_step, jit_compile=True, reduce_retracing=True)
        outputs = self.distribute_strategy.run(run_step, args=(batch,))
        outputs = reduce_per_replica(outputs, self.distribute_strategy, reduction=self.distribute_reduction_method)
        return outputs

    def shard_test_function(dataset, total_shards, shard_idx):
        local_unweighted_metrics, local_weighted_metrics = ([], [])
        with tf_utils.with_metric_local_vars_scope():
            for metric in self.compiled_metrics.unweighted_metrics:
                if metric is not None:
                    local_unweighted_metrics.append(base_metric.clone_metric(metric))
            for metric in self.compiled_metrics.weighted_metrics:
                if metric is not None:
                    local_weighted_metrics.append(base_metric.clone_metric(metric))
            local_loss = compile_utils.LossesContainer.from_config(self.compiled_loss.get_config())
        dataset = input_ops.auto_shard_dataset(dataset, total_shards, shard_idx)
        iterator = iter(dataset)
        with distribute_utils.cache_variable_reads():
            for batch in iterator:
                x, y, y_pred, sample_weight = step_function(batch)
                for weighted_metric in local_weighted_metrics:
                    weighted_metric.update_state(y, y_pred, sample_weight)
                for unweighted_metric in local_unweighted_metrics:
                    unweighted_metric.update_state(y, y_pred)
                local_loss(y, y_pred, sample_weight)
        local_metrics = local_unweighted_metrics + local_weighted_metrics + local_loss.metrics
        outputs = {metric.name: metric.weights for metric in local_metrics}
        with tf.control_dependencies(_minimum_control_deps(outputs)):
            self._test_counter.assign_add(1)
        return outputs
    if not self.run_eagerly:
        shard_test_function = tf.function(shard_test_function, reduce_retracing=True)
    self._shard_test_function = lambda *args: self._cluster_coordinator.schedule(shard_test_function, args=args)
    return self._shard_test_function