import logging
import operator
import os
import shutil
import sys
from itertools import chain
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K  # noqa: N812
import wandb
from wandb.sdk.integration_utils.data_logging import ValidationDataLogger
from wandb.sdk.lib.deprecate import Deprecated, deprecate
from wandb.util import add_import_hook
class _CustomOptimizer(_custom_optimizer_parent_class):

    def __init__(self):
        super().__init__(name='CustomOptimizer')
        self._resource_apply_dense = tf.function(self._resource_apply_dense)
        self._resource_apply_sparse = tf.function(self._resource_apply_sparse)

    def _resource_apply_dense(self, grad, var):
        var.assign(grad)

    def _resource_apply_sparse(self, grad, var, indices):
        pass

    def get_config(self):
        return super().get_config()