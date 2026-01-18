from typing import List
import pandas as pd
from fugue import FugueWorkflow
from pytest import raises
from tune import optimize_noniterative, suggest_for_noniterative_objective
from tune.concepts.dataset import TuneDatasetBuilder
from tune.concepts.flow import Monitor
from tune.concepts.space import Grid, Space
from tune.constants import TUNE_REPORT, TUNE_REPORT_METRIC
from tune.exceptions import TuneInterrupted
from tune.noniterative.convert import to_noniterative_objective
from tune.noniterative.stopper import n_samples
def test_study(tmpdir):
    space = Space(a=Grid(-2, 0, 1))
    input_df = pd.DataFrame([[0, 1], [1, 1], [0, 2]], columns=['a', 'b'])
    dag = FugueWorkflow()
    monitor = M()
    builder = TuneDatasetBuilder(space, str(tmpdir)).add_df('b', dag.df(input_df))
    dataset = builder.build(dag, 1)
    for distributed in [True, False, None]:
        result = optimize_noniterative(objective=objective, dataset=dataset, distributed=distributed)
        result.result()[[TUNE_REPORT, TUNE_REPORT_METRIC]].output(assert_metric, params=dict(metrics=[3.0, 4.0, 7.0]))
        result.result(2)[[TUNE_REPORT, TUNE_REPORT_METRIC]].output(assert_metric, params=dict(metrics=[3.0, 4.0]))
        result = optimize_noniterative(objective=to_noniterative_objective(objective, min_better=False), dataset=dataset, distributed=distributed)
        result.result()[[TUNE_REPORT, TUNE_REPORT_METRIC]].output(assert_metric, params=dict(metrics=[-7.0, -4.0, -3.0]))
        result.result(2)[[TUNE_REPORT, TUNE_REPORT_METRIC]].output(assert_metric, params=dict(metrics=[-7.0, -4.0]))
    builder = TuneDatasetBuilder(space, str(tmpdir)).add_df('b', dag.df(input_df).partition_by('a'))
    dataset = builder.build(dag, 1)
    for distributed in [True, False, None]:
        result = optimize_noniterative(objective=to_noniterative_objective(objective), dataset=dataset, distributed=distributed, monitor=monitor)
        result.result()[[TUNE_REPORT, TUNE_REPORT_METRIC]].output(assert_metric, params=dict(metrics=[2.0, 3.0, 6.0, 1.0, 2.0, 5.0]))
        result.result(1)[[TUNE_REPORT, TUNE_REPORT_METRIC]].output(assert_metric, params=dict(metrics=[1.0, 2.0]))
    dag.run()
    assert 3 * 3 * 2 == len(monitor._reports)