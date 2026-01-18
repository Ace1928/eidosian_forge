import pandas as pd
from tests.tune_tensorflow.mock import MockSpec
from tune import Trial, validate_iterative_objective
from tune_tensorflow import keras_space
from tune_tensorflow.objective import KerasObjective
from tune_tensorflow.utils import _TYPE_DICT
def test_spec():
    spec = MockSpec(dict(l1=16, l2=16), {})
    metric = spec.compute_sort_metric(epochs=10)
    assert metric < 15
    spec = MockSpec(dict(l1=16, l2=16), {'x': pd.DataFrame([[0]], columns=['a'])})
    metric = spec.compute_sort_metric(epochs=10)
    assert metric < 15
    assert isinstance(spec.dfs['x'], pd.DataFrame)