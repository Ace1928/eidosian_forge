from typing import Callable, Dict, Optional, Tuple, Union, TYPE_CHECKING
from copy import deepcopy
import logging
import numpy as np
import pandas as pd
from ray.tune import TuneError
from ray.tune.experiment import Trial
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pbt import _PBTTrialState
from ray.tune.utils.util import flatten_dict, unflatten_dict
from ray.util.debug import log_once
Gets new config for trial by exploring trial_to_clone's config using
        Bayesian Optimization (BO) to choose the hyperparameter values to explore.

        Overrides `PopulationBasedTraining._get_new_config`.

        Args:
            trial: The current trial that decided to exploit trial_to_clone.
            trial_to_clone: The top-performing trial with a hyperparameter config
                that the current trial will explore.

        Returns:
            new_config: New hyperparameter configuration (after BO).
            operations: Empty dict since PB2 doesn't explore in easily labeled ways
                like PBT does.
        