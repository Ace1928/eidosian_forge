import contextlib
import copy
import gc
import math
import os
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src import errors
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.backend import config
from keras_tuner.src.backend import keras
from keras_tuner.src.engine import base_tuner
from keras_tuner.src.engine import tuner_utils
def _override_compile_args(self, model):
    with maybe_distribute(self.distribution_strategy):
        if self.optimizer or self.loss or self.metrics:
            compile_kwargs = {'optimizer': model.optimizer, 'loss': model.loss, 'metrics': self._filter_metrics(model.metrics)}
            if self.loss:
                compile_kwargs['loss'] = self.loss
            if self.optimizer:
                optimizer = self.optimizer if isinstance(self.optimizer, str) else keras.optimizers.deserialize(keras.optimizers.serialize(self.optimizer))
                compile_kwargs['optimizer'] = optimizer
            if self.metrics:
                compile_kwargs['metrics'] = self.metrics
            model.compile(**compile_kwargs)