import json
from functools import partial, update_wrapper
from typing import Any, Dict, List
import numpy as np
import xgboost as xgb
import xgboost.testing as tm
def train_result(param: Dict[str, Any], dmat: xgb.DMatrix, num_rounds: int) -> Dict[str, Any]:
    """Get training result from parameters and data."""
    result: Dict[str, Any] = {}
    booster = xgb.train(param, dmat, num_rounds, evals=[(dmat, 'train')], verbose_eval=False, evals_result=result)
    assert booster.num_features() == dmat.num_col()
    assert booster.num_boosted_rounds() == num_rounds
    assert booster.feature_names == dmat.feature_names
    assert booster.feature_types == dmat.feature_types
    return result