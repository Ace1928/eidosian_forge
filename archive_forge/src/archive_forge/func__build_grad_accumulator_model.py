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
def _build_grad_accumulator_model(self):
    inputs = self.model.inputs
    outputs = self.model(inputs)
    grad_acc_model = tf.keras.models.Model(inputs, outputs)
    grad_acc_model.compile(loss=self.model.loss, optimizer=_CustomOptimizer())
    grad_acc_model._wandb_internal_model = True
    self._grad_accumulator_model = grad_acc_model
    self._grad_accumulator_callback = _GradAccumulatorCallback()