import copy
from time import sleep
import numpy as np
import pandas as pd
from tune.concepts.flow import (
from tune.concepts.space import Rand, TuningParametersTemplate
import cloudpickle
def test_trial_report():
    trial = Trial('abc', {'a': Rand(3, 4)}, {'b': 2})
    report = copy.copy(TrialReport(trial, metric=np.float(0.1), params={'c': Rand(1, 2)}, metadata={'d': 4}, cost=2.0))
    assert trial is report.trial
    report = cloudpickle.loads(cloudpickle.dumps(report))
    assert 0.1 == report.metric
    assert type(report.metric) == float
    assert {'c': Rand(1, 2)} == report.params
    assert {'d': 4} == report.metadata
    assert 2.0 == report.cost
    assert 0 == report.rung
    assert 0.1 == report.sort_metric
    report = copy.deepcopy(TrialReport(trial, metric=np.float(0.111), cost=2.0, rung=4, sort_metric=1.23))
    assert trial is report.trial
    report = cloudpickle.loads(cloudpickle.dumps(report))
    assert 0.111 == report.metric
    assert type(report.metric) == float
    assert {'a': Rand(3, 4)} == report.params
    assert {} == report.metadata
    assert 2.0 == report.cost
    assert 4 == report.rung
    r1 = report.generate_sort_metric(True, 2)
    r2 = report.generate_sort_metric(False, 1)
    r3 = report.with_sort_metric(0.234)
    assert 1.23 == report.sort_metric
    assert 0.11 == r1.sort_metric
    assert -0.1 == r2.sort_metric
    assert 0.234 == r3.sort_metric
    ts = report.log_time
    sleep(0.1)
    report = cloudpickle.loads(cloudpickle.dumps(report))
    nr = report.reset_log_time()
    assert nr.log_time > report.log_time
    assert report.log_time == ts
    assert trial.trial_id == report.trial_id
    assert 0.111 == report.metric
    assert type(report.metric) == float
    assert {'a': Rand(3, 4)} == report.params
    assert {} == report.metadata
    assert 2.0 == report.cost
    assert 3.0 == report.with_cost(3.0).cost
    assert 5 == report.with_rung(5).rung
    td = trial.with_dfs({'a': pd.DataFrame})
    report = TrialReport(td, metric=np.float(0.1))
    assert 0 == len(report.trial.dfs)