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
def test_study_with_exceptions(tmpdir):
    space = Space(a=Grid(-2, 0, 1))
    input_df = pd.DataFrame([[0, 1], [1, 1], [0, 2]], columns=['a', 'b'])
    with raises(ValueError):
        suggest_for_noniterative_objective(objective=objective4, space=space, df=input_df, df_name='b', temp_path=str(tmpdir))
    with raises(ValueError):
        monitor = M()
        suggest_for_noniterative_objective(objective=objective4, space=space, df=input_df, df_name='b', stopper=n_samples(2), monitor=monitor, temp_path=str(tmpdir))