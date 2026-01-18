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
def test_study_with_stopper(tmpdir):
    space = Space(a=Grid(-2, 0, 1))
    input_df = pd.DataFrame([[0, 1], [1, 1], [0, 2]], columns=['a', 'b'])
    result = suggest_for_noniterative_objective(objective=objective, space=space, df=input_df, df_name='b', stopper=n_samples(2), top_n=0, shuffle_candidates=False, temp_path=str(tmpdir))
    assert [3.0, 7.0] == [x.metric for x in result]
    monitor = M()
    result = suggest_for_noniterative_objective(objective=objective, space=space, df=input_df, df_name='b', stopper=n_samples(2), monitor=monitor, top_n=0, shuffle_candidates=False, temp_path=str(tmpdir))
    assert [3.0, 7.0] == [x.metric for x in result]
    assert 2 == len(monitor._reports)
    monitor = M()
    result = suggest_for_noniterative_objective(objective=objective3, space=space, df=input_df, df_name='b', stopper=n_samples(2), monitor=monitor, top_n=0, shuffle_candidates=False, temp_path=str(tmpdir))
    assert [3.0, 4.0] == [x.metric for x in result]
    assert 2 == len(monitor._reports)