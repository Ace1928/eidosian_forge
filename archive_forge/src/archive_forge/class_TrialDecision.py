from typing import Any, Callable, Dict, Optional
from tune.concepts.flow.report import TrialReport
from tune.concepts.flow.trial import Trial
class TrialDecision:

    def __init__(self, report: TrialReport, budget: float, should_checkpoint: bool, reason: str='', metadata: Optional[Dict[str, Any]]=None):
        self._report = report
        self._budget = budget
        self._should_checkpoint = should_checkpoint
        self._reason = reason
        self._metadata = metadata or {}

    def __repr__(self) -> str:
        return repr(dict(report=self._report, budget=self._budget, should_checkpoint=self._should_checkpoint, reason=self._reason, metadata=self._metadata))

    def __copy__(self) -> 'TrialDecision':
        return self

    def __deepcopy__(self, memo: Any) -> 'TrialDecision':
        return self

    @property
    def report(self) -> TrialReport:
        return self._report

    @property
    def trial(self) -> Trial:
        return self.report.trial

    @property
    def trial_id(self) -> str:
        return self.trial.trial_id

    @property
    def budget(self) -> float:
        return self._budget

    @property
    def should_stop(self) -> bool:
        return self.budget <= 0

    @property
    def should_checkpoint(self) -> bool:
        return self._should_checkpoint

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata