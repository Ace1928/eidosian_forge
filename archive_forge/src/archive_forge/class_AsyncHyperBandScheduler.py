import logging
from typing import Dict, Optional, Union, TYPE_CHECKING
import numpy as np
import pickle
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.util import PublicAPI
@PublicAPI
class AsyncHyperBandScheduler(FIFOScheduler):
    """Implements the Async Successive Halving.

    This should provide similar theoretical performance as HyperBand but
    avoid straggler issues that HyperBand faces. One implementation detail
    is when using multiple brackets, trial allocation to bracket is done
    randomly with over a softmax probability.

    See https://arxiv.org/abs/1810.05934

    Args:
        time_attr: A training result attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `training_iteration` as a measure of progress, the only requirement
            is that the attribute should increase monotonically.
        metric: The training result objective value attribute. Stopping
            procedures will use this attribute. If None but a mode was passed,
            the `ray.tune.result.DEFAULT_METRIC` will be used per default.
        mode: One of {min, max}. Determines whether objective is
            minimizing or maximizing the metric attribute.
        max_t: max time units per trial. Trials will be stopped after
            max_t time units (determined by time_attr) have passed.
        grace_period: Only stop trials at least this old in time.
            The units are the same as the attribute named by `time_attr`.
        reduction_factor: Used to set halving rate and amount. This
            is simply a unit-less scalar.
        brackets: Number of brackets. Each bracket has a different
            halving rate, specified by the reduction factor.
        stop_last_trials: Whether to terminate the trials after
            reaching max_t. Defaults to True.
    """

    def __init__(self, time_attr: str='training_iteration', metric: Optional[str]=None, mode: Optional[str]=None, max_t: int=100, grace_period: int=1, reduction_factor: float=4, brackets: int=1, stop_last_trials: bool=True):
        assert max_t > 0, 'Max (time_attr) not valid!'
        assert max_t >= grace_period, 'grace_period must be <= max_t!'
        assert grace_period > 0, 'grace_period must be positive!'
        assert reduction_factor > 1, 'Reduction Factor not valid!'
        assert brackets > 0, 'brackets must be positive!'
        if mode:
            assert mode in ['min', 'max'], "`mode` must be 'min' or 'max'!"
        super().__init__()
        self._reduction_factor = reduction_factor
        self._max_t = max_t
        self._trial_info = {}
        self._brackets = [_Bracket(grace_period, max_t, reduction_factor, s, stop_last_trials=stop_last_trials) for s in range(brackets)]
        self._counter = 0
        self._num_stopped = 0
        self._metric = metric
        self._mode = mode
        self._metric_op = None
        if self._mode == 'max':
            self._metric_op = 1.0
        elif self._mode == 'min':
            self._metric_op = -1.0
        self._time_attr = time_attr
        self._stop_last_trials = stop_last_trials

    def set_search_properties(self, metric: Optional[str], mode: Optional[str], **spec) -> bool:
        if self._metric and metric:
            return False
        if self._mode and mode:
            return False
        if metric:
            self._metric = metric
        if mode:
            self._mode = mode
        if self._mode == 'max':
            self._metric_op = 1.0
        elif self._mode == 'min':
            self._metric_op = -1.0
        if self._metric is None and self._mode:
            self._metric = DEFAULT_METRIC
        return True

    def on_trial_add(self, tune_controller: 'TuneController', trial: Trial):
        if not self._metric or not self._metric_op:
            raise ValueError('{} has been instantiated without a valid `metric` ({}) or `mode` ({}) parameter. Either pass these parameters when instantiating the scheduler, or pass them as parameters to `tune.TuneConfig()`'.format(self.__class__.__name__, self._metric, self._mode))
        sizes = np.array([len(b._rungs) for b in self._brackets])
        probs = np.e ** (sizes - sizes.max())
        normalized = probs / probs.sum()
        idx = np.random.choice(len(self._brackets), p=normalized)
        self._trial_info[trial.trial_id] = self._brackets[idx]

    def on_trial_result(self, tune_controller: 'TuneController', trial: Trial, result: Dict) -> str:
        action = TrialScheduler.CONTINUE
        if self._time_attr not in result or self._metric not in result:
            return action
        if result[self._time_attr] >= self._max_t and self._stop_last_trials:
            action = TrialScheduler.STOP
        else:
            bracket = self._trial_info[trial.trial_id]
            action = bracket.on_result(trial, result[self._time_attr], self._metric_op * result[self._metric])
        if action == TrialScheduler.STOP:
            self._num_stopped += 1
        return action

    def on_trial_complete(self, tune_controller: 'TuneController', trial: Trial, result: Dict):
        if self._time_attr not in result or self._metric not in result:
            return
        bracket = self._trial_info[trial.trial_id]
        bracket.on_result(trial, result[self._time_attr], self._metric_op * result[self._metric])
        del self._trial_info[trial.trial_id]

    def on_trial_remove(self, tune_controller: 'TuneController', trial: Trial):
        del self._trial_info[trial.trial_id]

    def debug_string(self) -> str:
        out = 'Using AsyncHyperBand: num_stopped={}'.format(self._num_stopped)
        out += '\n' + '\n'.join([b.debug_str() for b in self._brackets])
        return out

    def save(self, checkpoint_path: str):
        save_object = self.__dict__
        with open(checkpoint_path, 'wb') as outputFile:
            pickle.dump(save_object, outputFile)

    def restore(self, checkpoint_path: str):
        with open(checkpoint_path, 'rb') as inputFile:
            save_object = pickle.load(inputFile)
        self.__dict__.update(save_object)