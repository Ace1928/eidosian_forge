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
def _log_gradients(self):
    og_level = tf_logger.level
    tf_logger.setLevel('ERROR')
    self._grad_accumulator_model.fit(self._training_data_x, self._training_data_y, verbose=0, callbacks=[self._grad_accumulator_callback])
    tf_logger.setLevel(og_level)
    weights = self.model.trainable_weights
    grads = self._grad_accumulator_callback.grads
    metrics = {}
    for weight, grad in zip(weights, grads):
        metrics['gradients/' + weight.name.split(':')[0] + '.gradient'] = wandb.Histogram(grad)
    return metrics