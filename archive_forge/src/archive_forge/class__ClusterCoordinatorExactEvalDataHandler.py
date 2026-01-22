import abc
import contextlib
import functools
import itertools
import math
import random
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.distribute import distributed_training_utils
from keras.src.engine import training_utils
from keras.src.utils import data_utils
from keras.src.utils import dataset_creator
from keras.src.utils import tf_utils
from tensorflow.python.distribute.input_lib import (
from tensorflow.python.eager import context
from tensorflow.python.framework import type_spec
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.data.ops import (
from tensorflow.python.data.ops import from_generator_op
from tensorflow.python.data.ops import range_op
from tensorflow.python.data.ops import from_tensors_op
from tensorflow.python.data.ops import from_tensor_slices_op
class _ClusterCoordinatorExactEvalDataHandler(_ClusterCoordinatorDataHandler):

    def __init__(self, x, y=None, **kwargs):
        super().__init__(x=x, **kwargs)
        self._total_shards = kwargs.get('pss_evaluation_shards')

    def _warn_if_not_file_shardable(self, dataset):
        cur_dataset = dataset
        while hasattr(cur_dataset, '_input_dataset'):
            cur_dataset = cur_dataset._input_dataset
        if type(cur_dataset) in UNSHARDABLE_DATASET_TYPES:
            logging.warning('Found source dataset of type {}. This type is not efficiently shardable, so exact evaluation may be slower than inexact evaluation. Try converting to a TFRecord or other file-based dataset if performance is a concern.'.format(type(cur_dataset)))

    def _configure_dataset_and_inferred_steps(self, strategy, x, steps_per_epoch, class_weight, distribute):
        if isinstance(x, dataset_creator.DatasetCreator):

            def per_worker_dataset_fn():
                ddf = strategy.distribute_datasets_from_function(x, options=x.input_options)
                return ddf
            coordinator = self._model._cluster_coordinator
            self._dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
            logging.info('dataset element spec: %r', self._dataset.element_spec)
            self._dataset = self._dataset.build()
        else:
            if not _is_distributed_dataset(x):
                self._warn_if_not_file_shardable(x)
                x = strategy.experimental_distribute_dataset(x)
            coordinator = self._model._cluster_coordinator
            self._dataset = coordinator.create_per_worker_dataset(x)
            self._dataset = self._dataset.build()
        if steps_per_epoch == -1:
            self._inferred_steps = None
            self._log_indefinite_training_warning()
        else:
            self._inferred_steps = steps_per_epoch

    def enumerate_epochs(self):
        """Yields `(epoch, dataset)`."""
        for epoch in range(self._initial_epoch, self._epochs):
            yield (epoch, self._dataset)
            self._adapter.on_epoch_end()

    def steps(self):
        """Yields steps for the current epoch."""
        for step in range(self._total_shards):
            yield step