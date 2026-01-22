from datetime import datetime
from typing import Any, Callable, Dict, List, Set
from triad import SerializableRLock
from triad.utils.convert import to_timedelta
from tune.concepts.flow import (
class NonIterativeStopperCombiner(NonIterativeStopper):

    def __init__(self, left: NonIterativeStopper, right: NonIterativeStopper, is_and: bool):
        super().__init__()
        assert not left.updated, "can't reuse updated stopper"
        assert not right.updated, "can't reuse updated stopper"
        self._left = left
        self._right = right
        self._is_and = is_and

    def should_stop(self, trial: Trial) -> bool:
        if self._is_and:
            return self._left.should_stop(trial) and self._right.should_stop(trial)
        else:
            return self._left.should_stop(trial) or self._right.should_stop(trial)

    def on_report(self, report: TrialReport) -> bool:
        self.monitor.on_report(report)
        left = self._left.on_report(report)
        right = self._right.on_report(report)
        return left or right

    def get_reports(self, trial: Trial) -> List[TrialReport]:
        raise NotImplementedError