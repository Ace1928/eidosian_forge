from copy import copy
from typing import Any, Callable, Dict, Iterable, Optional
from fugue import ArrayDataFrame, DataFrame, ExecutionEngine
from triad import assert_or_throw
from tune._utils import run_monitored_process
from tune.concepts.dataset import StudyResult, TuneDataset, _get_trials_from_row
from tune.concepts.flow import RemoteTrialJudge, TrialCallback, TrialJudge, TrialReport
from tune.concepts.flow.judge import Monitor, NoOpTrailJudge
from tune.constants import TUNE_REPORT_ADD_SCHEMA, TUNE_STOPPER_DEFAULT_CHECK_INTERVAL
from tune.exceptions import TuneCompileError, TuneInterrupted
from tune.noniterative.objective import (
from tune.noniterative.stopper import NonIterativeStopper
def _local_process_trial(self, row: Dict[str, Any], idx: int, logger: Any) -> TrialReport:
    trial = list(_get_trials_from_row(row))[idx]
    objective = copy(self._objective)
    return self._optimizer.run(objective, trial, logger=logger)