import logging
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, List, Union
from ray.air.constants import TRAINING_ITERATION
from ray.tune.logger.logger import LoggerCallback
from ray.tune.result import (
from ray.tune.utils import flatten_dict
from ray.util.annotations import PublicAPI
def _create_run(self, trial: 'Trial') -> Run:
    """Initializes an Aim Run object for a given trial.

        Args:
            trial: The Tune trial that aim will track as a Run.

        Returns:
            Run: The created aim run for a specific trial.
        """
    experiment_dir = trial.local_experiment_path
    run = Run(repo=self._repo_path or experiment_dir, experiment=self._experiment_name or trial.experiment_dir_name, **self._aim_run_kwargs)
    run['trial_id'] = trial.trial_id
    run['trial_log_dir'] = trial.path
    trial_ip = trial.get_ray_actor_ip()
    if trial_ip:
        run['trial_ip'] = trial_ip
    return run