import os
import tempfile
from typing import Callable, List, Optional
from uuid import uuid4
import cloudpickle
from fs.base import FS as FSBase
from triad import FileSystem
from tune.concepts.checkpoint import Checkpoint
from tune.concepts.flow import Monitor, Trial, TrialDecision, TrialJudge, TrialReport
def validate_iterative_objective(func: IterativeObjectiveFunc, trial: Trial, budgets: List[float], validator: Callable[[List[TrialReport]], None], continuous: bool=False, checkpoint_path: str='', monitor: Optional[Monitor]=None) -> None:
    path = checkpoint_path if checkpoint_path != '' else tempfile.gettempdir()
    _basefs = FileSystem()
    basefs = _basefs.makedirs(os.path.join(path, str(uuid4())), recreate=True)
    j = _Validator(monitor, budgets, continuous=continuous)
    if continuous:
        f = cloudpickle.loads(cloudpickle.dumps(func)).copy()
        f.run(trial, j, checkpoint_basedir_fs=basefs)
    else:
        for _ in budgets:
            f = cloudpickle.loads(cloudpickle.dumps(func)).copy()
            f.run(trial, j, checkpoint_basedir_fs=basefs)
    validator(j.reports)