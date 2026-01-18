import copy
from time import sleep
import numpy as np
import pandas as pd
from tune.concepts.flow import (
from tune.concepts.space import Rand, TuningParametersTemplate
import cloudpickle
def test_trial_serialization():
    p = {'a': 1, 'b': Rand(1, 2)}
    trial = Trial('abc', p, {}, keys=['x', 'y'], dfs={'v': ''})
    t = cloudpickle.loads(cloudpickle.dumps(trial))
    assert isinstance(t.params, TuningParametersTemplate)
    assert ['x', 'y'] == t.keys
    assert t.trial_id == trial.trial_id
    assert {'v': ''} == t.dfs