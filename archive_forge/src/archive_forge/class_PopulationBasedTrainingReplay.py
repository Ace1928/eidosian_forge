import copy
import json
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from ray.air.constants import TRAINING_ITERATION
from ray.train import Checkpoint
from ray.train._internal.session import _TrainingResult, _FutureTrainingResult
from ray.tune.error import TuneError
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search import SearchGenerator
from ray.tune.utils.util import SafeFallbackEncoder
from ray.tune.search.sample import Domain, Function
from ray.tune.schedulers import FIFOScheduler, TrialScheduler
from ray.tune.search.variant_generator import format_vars
from ray.tune.experiment import Trial
from ray.util import PublicAPI
from ray.util.debug import log_once
@PublicAPI
class PopulationBasedTrainingReplay(FIFOScheduler):
    """Replays a Population Based Training run.

    Population Based Training does not return a single hyperparameter
    configuration, but rather a schedule of configurations. For instance,
    PBT might discover that a larger learning rate leads to good results
    in the first training iterations, but that a smaller learning rate
    is preferable later.

    This scheduler enables replaying these parameter schedules from
    a finished PBT run. This requires that population based training has
    been run with ``log_config=True``, which is the default setting.

    The scheduler will only accept and train a single trial. It will
    start with the initial config of the existing trial and update the
    config according to the schedule.

    Args:
        policy_file: The PBT policy file. Usually this is
            stored in ``~/ray_results/experiment_name/pbt_policy_xxx.txt``
            where ``xxx`` is the trial ID.

    Example:

    .. code-block:: python

        # Replaying a result from ray.tune.examples.pbt_convnet_example
        from ray import train, tune

        from ray.tune.examples.pbt_convnet_example import PytorchTrainable
        from ray.tune.schedulers import PopulationBasedTrainingReplay

        replay = PopulationBasedTrainingReplay(
            "~/ray_results/pbt_test/pbt_policy_XXXXX_00001.txt")

        tuner = tune.Tuner(
            PytorchTrainable,
            run_config=train.RunConfig(
                stop={"training_iteration": 100}
            ),
            tune_config=tune.TuneConfig(
                scheduler=replay,
            ),
        )
        tuner.fit()


    """

    def __init__(self, policy_file: str):
        policy_file = Path(policy_file).expanduser()
        if not policy_file.exists():
            raise ValueError('Policy file not found: {}'.format(policy_file.as_posix()))
        self.policy_file = policy_file.as_posix()
        initial_config, self._policy = self._load_policy(self.policy_file)
        self.experiment_tag = 'replay_{}'.format(os.path.basename(self.policy_file))
        self.config = initial_config
        self.current_config = self.config
        self._trial = None
        self._current_step = 0
        self._num_perturbations = 0
        self._policy_iter = iter(self._policy)
        self._next_policy = next(self._policy_iter, None)

    def _load_policy(self, policy_file: str) -> Tuple[Dict, List[Tuple[int, Dict]]]:
        raw_policy = []
        with open(policy_file, 'rt') as fp:
            for row in fp.readlines():
                try:
                    parsed_row = json.loads(row)
                except json.JSONDecodeError:
                    raise ValueError('Could not read PBT policy file: {}.'.format(policy_file)) from None
                raw_policy.append(tuple(parsed_row))
        policy = []
        last_new_tag = None
        last_old_conf = None
        for old_tag, new_tag, old_step, new_step, old_conf, new_conf in reversed(raw_policy):
            if last_new_tag and old_tag != last_new_tag:
                break
            last_new_tag = new_tag
            last_old_conf = old_conf
            policy.append((new_step, new_conf))
        return (last_old_conf, list(reversed(policy)))

    def on_trial_add(self, tune_controller: 'TuneController', trial: Trial):
        if self._trial:
            raise ValueError('More than one trial added to PBT replay run. This means the same schedule will be trained multiple times. Do you want to set `n_samples=1`?')
        self._trial = trial
        if self._trial.config and self._policy:
            logger.warning('Trial was initialized with a config, which was overwritten. Did you start the PBT replay with a `config` parameter?')
        elif self._trial.config and (not self._policy):
            self.config = self._trial.config
        elif not self._trial.config and (not self._policy):
            raise ValueError('No replay policy found and trial initialized without a valid config. Either pass a `config` argument to `tune.Tuner()`or consider not using PBT replay for this run.')
        self._trial.set_config(self.config)

    def on_trial_result(self, tune_controller: 'TuneController', trial: Trial, result: Dict) -> str:
        if TRAINING_ITERATION not in result:
            return TrialScheduler.CONTINUE
        if not self._next_policy:
            return TrialScheduler.CONTINUE
        step = result[TRAINING_ITERATION]
        self._current_step = step
        change_at, new_config = self._next_policy
        if step < change_at:
            return TrialScheduler.CONTINUE
        logger.info('Population Based Training replay is now at step {}. Configuration will be changed to {}.'.format(step, new_config))
        result = tune_controller._schedule_trial_save(trial, result=result)
        training_result = result.resolve()
        trial.run_metadata.checkpoint_manager._latest_checkpoint_result = training_result
        new_tag = _make_experiment_tag(self.experiment_tag, new_config, new_config)
        tune_controller.pause_trial(trial, should_checkpoint=False)
        trial.set_experiment_tag(new_tag)
        trial.set_config(new_config)
        self.current_config = new_config
        self._num_perturbations += 1
        self._next_policy = next(self._policy_iter, None)
        return TrialScheduler.NOOP

    def debug_string(self) -> str:
        return 'PopulationBasedTraining replay: Step {}, perturb {}'.format(self._current_step, self._num_perturbations)