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
class _GradAccumulatorCallback(tf.keras.callbacks.Callback):
    """Accumulates gradients during a fit() call when used in conjunction with the CustomOptimizer above."""

    def set_model(self, model):
        super().set_model(model)
        self.og_weights = model.get_weights()
        self.grads = [np.zeros(tuple(w.shape)) for w in model.trainable_weights]

    def on_batch_end(self, batch, logs=None):
        for g, w in zip(self.grads, self.model.trainable_weights):
            g += w.numpy()
        self.model.set_weights(self.og_weights)

    def get_grads(self):
        return [g.copy() for g in self.grads]