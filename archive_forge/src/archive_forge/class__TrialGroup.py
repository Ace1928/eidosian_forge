import copy
import logging
from typing import Dict, List, Optional
import numpy as np
from ray.tune.search import Searcher
from ray.tune.search.util import _set_search_properties_backwards_compatible
from ray.util import PublicAPI
class _TrialGroup:
    """Internal class for grouping trials of same parameters.

    This is used when repeating trials for reducing training variance.

    Args:
        primary_trial_id: Trial ID of the "primary trial".
            This trial is the one that the Searcher is aware of.
        config: Suggested configuration shared across all trials
            in the trial group.
        max_trials: Max number of trials to execute within this group.

    """

    def __init__(self, primary_trial_id: str, config: Dict, max_trials: int=1):
        assert type(config) is dict, 'config is not a dict, got {}'.format(config)
        self.primary_trial_id = primary_trial_id
        self.config = config
        self._trials = {primary_trial_id: None}
        self.max_trials = max_trials

    def add(self, trial_id: str):
        assert len(self._trials) < self.max_trials
        self._trials.setdefault(trial_id, None)

    def full(self) -> bool:
        return len(self._trials) == self.max_trials

    def report(self, trial_id: str, score: float):
        assert trial_id in self._trials
        if score is None:
            raise ValueError('Internal Error: Score cannot be None.')
        self._trials[trial_id] = score

    def finished_reporting(self) -> bool:
        return None not in self._trials.values() and len(self._trials) == self.max_trials

    def scores(self) -> List[Optional[float]]:
        return list(self._trials.values())

    def count(self) -> int:
        return len(self._trials)