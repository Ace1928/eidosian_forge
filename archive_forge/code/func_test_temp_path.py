import pandas as pd
from fugue.workflow.workflow import FugueWorkflow
from pytest import raises
from tune import Space, MetricLogger
from tune.api.factory import (
from tune.concepts.dataset import TuneDataset
from tune.concepts.flow.judge import Monitor
from tune.exceptions import TuneCompileError
from tune.iterative.objective import IterativeObjectiveFunc
from tune.noniterative.convert import to_noniterative_objective
from tune.noniterative.objective import (
from tune.noniterative.stopper import NonIterativeStopper
from tune_optuna.optimizer import OptunaLocalOptimizer
def test_temp_path():
    factory = TuneObjectFactory()
    with raises(TuneCompileError):
        factory.get_path_or_temp('')
    assert '/x' == factory.get_path_or_temp('/x')
    factory.set_temp_path('/tmp')
    assert '/x' == factory.get_path_or_temp('/x')
    assert '/tmp' == factory.get_path_or_temp('')