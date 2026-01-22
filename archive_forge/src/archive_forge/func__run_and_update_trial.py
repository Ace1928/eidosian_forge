import copy
import os
import traceback
import warnings
from keras_tuner.src import backend
from keras_tuner.src import config as config_module
from keras_tuner.src import errors
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.distribute import utils as dist_utils
from keras_tuner.src.engine import hypermodel as hm_module
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import stateful
from keras_tuner.src.engine import trial as trial_module
from keras_tuner.src.engine import tuner_utils
def _run_and_update_trial(self, trial, *fit_args, **fit_kwargs):
    results = self.run_trial(trial, *fit_args, **fit_kwargs)
    if self.oracle.get_trial(trial.trial_id).metrics.exists(self.oracle.objective.name):
        warnings.warn("The use case of calling `self.oracle.update_trial(trial_id, metrics)` in `Tuner.run_trial()` to report the metrics is deprecated, and will be removed in the future.Please remove the call and do 'return metrics' in `Tuner.run_trial()` instead. ", DeprecationWarning, stacklevel=2)
        return
    (tuner_utils.validate_trial_results(results, self.oracle.objective, 'Tuner.run_trial()'),)
    self.oracle.update_trial(trial.trial_id, tuner_utils.convert_to_metrics_dict(results, self.oracle.objective), step=tuner_utils.get_best_step(results, self.oracle.objective))