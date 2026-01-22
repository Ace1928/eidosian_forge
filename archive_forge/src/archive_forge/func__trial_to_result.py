import os
import pandas as pd
import pyarrow
from typing import Optional, Union
from ray.air.result import Result
from ray.cloudpickle import cloudpickle
from ray.exceptions import RayTaskError
from ray.tune.analysis import ExperimentAnalysis
from ray.tune.error import TuneError
from ray.tune.experiment import Trial
from ray.util import PublicAPI
def _trial_to_result(self, trial: Trial) -> Result:
    cpm = trial.run_metadata.checkpoint_manager
    checkpoint = None
    if cpm.latest_checkpoint_result:
        checkpoint = cpm.latest_checkpoint_result.checkpoint
    best_checkpoint_results = cpm.best_checkpoint_results
    best_checkpoints = [(checkpoint_result.checkpoint, checkpoint_result.metrics) for checkpoint_result in best_checkpoint_results]
    metrics_df = self._experiment_analysis.trial_dataframes.get(trial.trial_id)
    result = Result(checkpoint=checkpoint, metrics=trial.last_result.copy(), error=self._populate_exception(trial), path=trial.path, _storage_filesystem=self._experiment_analysis._fs, metrics_dataframe=metrics_df, best_checkpoints=best_checkpoints)
    return result