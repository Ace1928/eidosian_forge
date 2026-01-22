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
def _build_and_fit_model(self, trial, *args, **kwargs):
    """For AutoKeras to override.

        DO NOT REMOVE this function. AutoKeras overrides the function to tune
        tf.data preprocessing pipelines, preprocess the dataset to obtain
        the input shape before building the model, adapt preprocessing layers,
        and tune other fit_args and fit_kwargs.

        Args:
            trial: A `Trial` instance that contains the information needed to
                run this trial. `Hyperparameters` can be accessed via
                `trial.hyperparameters`.
            *args: Positional arguments passed by `search`.
            **kwargs: Keyword arguments passed by `search`.

        Returns:
            The fit history.
        """
    hp = trial.hyperparameters
    model = self._try_build(hp)
    results = self.hypermodel.fit(hp, model, *args, **kwargs)
    if backend.config.multi_backend():
        utils.save_json(self._get_build_config_fname(trial.trial_id), model.get_build_config())
    tuner_utils.validate_trial_results(results, self.oracle.objective, 'HyperModel.fit()')
    return results