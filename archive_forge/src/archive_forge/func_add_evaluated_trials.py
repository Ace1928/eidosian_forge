import copy
import glob
import logging
import os
import warnings
from typing import Dict, Optional, List, Union, Any, TYPE_CHECKING
from ray.air._internal.usage import tag_searcher
from ray.tune.search.util import _set_search_properties_backwards_compatible
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.debug import log_once
def add_evaluated_trials(self, trials_or_analysis: Union['Trial', List['Trial'], 'ExperimentAnalysis'], metric: str):
    """Pass results from trials that have been evaluated separately.

        This method allows for information from outside the
        suggest - on_trial_complete loop to be passed to the search
        algorithm.
        This functionality depends on the underlying search algorithm
        and may not be always available (same as ``add_evaluated_point``.)

        Args:
            trials_or_analysis: Trials to pass results form to the searcher.
            metric: Metric name reported by trials used for
                determining the objective value.

        """
    if self.add_evaluated_point == Searcher.add_evaluated_point:
        raise NotImplementedError
    from ray.tune.experiment import Trial
    from ray.tune.analysis import ExperimentAnalysis
    from ray.tune.result import DONE
    if isinstance(trials_or_analysis, (list, tuple)):
        trials = trials_or_analysis
    elif isinstance(trials_or_analysis, Trial):
        trials = [trials_or_analysis]
    elif isinstance(trials_or_analysis, ExperimentAnalysis):
        trials = trials_or_analysis.trials
    else:
        raise NotImplementedError(f'Expected input to be a `Trial`, a list of `Trial`s, or `ExperimentAnalysis`, got: {trials_or_analysis}')
    any_trial_had_metric = False

    def trial_to_points(trial: Trial) -> Dict[str, Any]:
        nonlocal any_trial_had_metric
        has_trial_been_pruned = trial.status == Trial.TERMINATED and (not trial.last_result.get(DONE, False))
        has_trial_finished = trial.status == Trial.TERMINATED and trial.last_result.get(DONE, False)
        if not any_trial_had_metric:
            any_trial_had_metric = metric in trial.last_result and has_trial_finished
        if Trial.TERMINATED and metric not in trial.last_result:
            return None
        return dict(parameters=trial.config, value=trial.last_result.get(metric, None), error=trial.status == Trial.ERROR, pruned=has_trial_been_pruned, intermediate_values=None)
    for trial in trials:
        kwargs = trial_to_points(trial)
        if kwargs:
            self.add_evaluated_point(**kwargs)
    if not any_trial_had_metric:
        warnings.warn('No completed trial returned the specified metric. Make sure the name you have passed is correct. ')