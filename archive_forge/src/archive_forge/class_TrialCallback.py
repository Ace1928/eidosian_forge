from typing import Any, Callable, Dict, Optional
from tune.concepts.flow.report import TrialReport
from tune.concepts.flow.trial import Trial
class TrialCallback:

    def __init__(self, judge: TrialJudge):
        self._judge = judge

    def entrypoint(self, name, kwargs: Dict[str, Any]) -> Any:
        if name == 'can_accept':
            return self._judge.can_accept(kwargs['trial'])
        if name == 'judge':
            return self._judge.judge(kwargs['report'])
        if name == 'get_budget':
            return self._judge.get_budget(kwargs['trial'], kwargs['rung'])
        raise NotImplementedError