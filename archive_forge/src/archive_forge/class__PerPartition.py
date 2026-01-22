from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from triad import SerializableRLock, to_uuid
from tune.concepts.flow import (
class _PerPartition:

    def __init__(self, parent: 'ASHAJudge', keys: List[Any]):
        self._keys = keys
        self._data: Dict[str, _PerTrial] = {}
        self._lock = SerializableRLock()
        self._parent = parent
        self._rungs: List[RungHeap] = [RungHeap(x[1]) for x in self._parent.schedule]
        self._active = True
        self._accepted_ids: Set[str] = set()

    def can_accept(self, trial: Trial) -> bool:
        with self._lock:
            if self._active:
                self._active = not self._parent._study_early_stop(self._keys, self._rungs)
                if self._active:
                    self._accepted_ids.add(trial.trial_id)
                    return True
            return trial.trial_id in self._accepted_ids

    def get_budget(self, trial: Trial, rung: int) -> float:
        if rung >= len(self._parent.schedule) or not self.can_accept(trial):
            return 0.0
        return self._parent.schedule[rung][0]

    def judge(self, report: TrialReport) -> TrialDecision:
        return self._get_judge(report.trial).judge(report)

    def _get_judge(self, trial: Trial) -> _PerTrial:
        key = trial.trial_id
        with self._lock:
            if key not in self._data:
                self._data[key] = _PerTrial(self)
            return self._data[key]