import collections
import warnings
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.keras.distribute import distributed_training_utils_v1
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.engine import training_arrays_v1
from tensorflow.python.keras.engine import training_distributed_v1
from tensorflow.python.keras.engine import training_eager_v1
from tensorflow.python.keras.engine import training_generator_v1
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def _select_training_loop(self, inputs):
    """Select training loop for fit/eval/predict based on the inputs."""
    if isinstance(inputs, (iterator_ops.Iterator, iterator_ops.IteratorBase)):
        raise ValueError('For performance reasons Keras `fit`, `evaluate` and`predict` accept tf.data `Datasets` as input but not iterators that have been manually generated from Datasets by users. Please directly pass in the original `Dataset` object instead of passing in `iter(dataset)`.')
    if self._distribution_strategy:
        if self._in_multi_worker_mode():
            return training_distributed_v1.DistributionMultiWorkerTrainingLoop(training_distributed_v1.DistributionSingleWorkerTrainingLoop())
        else:
            return training_distributed_v1.DistributionSingleWorkerTrainingLoop()
    if data_utils.is_generator_or_sequence(inputs):
        return training_generator_v1.GeneratorOrSequenceTrainingLoop()
    if training_utils_v1.is_eager_dataset_or_iterator(inputs):
        return training_generator_v1.EagerDatasetOrIteratorTrainingLoop()
    if self.run_eagerly:
        return training_generator_v1.GeneratorLikeTrainingLoop()
    else:
        return training_arrays_v1.ArrayLikeTrainingLoop()