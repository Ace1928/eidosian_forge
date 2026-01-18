import multiprocessing as mp
import numpy as np
import pytest
import ray
import xgboost
from sklearn.datasets import (
from sklearn.metrics import accuracy_score, mean_squared_error
import modin
import modin.experimental.xgboost as xgb
import modin.pandas as pd
from modin.config import Engine
from modin.experimental.sklearn.model_selection.train_test_split import train_test_split
@pytest.mark.parametrize('modin_type_y', [pd.DataFrame, pd.Series])
@pytest.mark.parametrize('num_actors', [1, num_cpus, None, modin.config.NPartitions.get() + 1])
@pytest.mark.parametrize('data', [(load_breast_cancer(), {'objective': 'binary:logistic', 'eval_metric': ['logloss', 'error']})], ids=['load_breast_cancer'])
def test_xgb_with_binary_classification_datasets(data, num_actors, modin_type_y):
    dataset, param = data
    num_round = 10
    X = dataset.data
    y = dataset.target
    xgb_dmatrix = xgboost.DMatrix(X, label=y)
    modin_X = pd.DataFrame(X)
    modin_y = modin_type_y(y)
    mxgb_dmatrix = xgb.DMatrix(modin_X, label=modin_y)
    evals_result_xgb = {}
    evals_result_mxgb = {}
    verbose_eval = False
    bst = xgboost.train(param, xgb_dmatrix, num_round, evals_result=evals_result_xgb, evals=[(xgb_dmatrix, 'train')], verbose_eval=verbose_eval)
    modin_bst = xgb.train(param, mxgb_dmatrix, num_round, evals_result=evals_result_mxgb, evals=[(mxgb_dmatrix, 'train')], num_actors=num_actors, verbose_eval=verbose_eval)
    for par in param['eval_metric']:
        assert len(evals_result_xgb['train'][par]) == len(evals_result_xgb['train'][par])
        for i in range(len(evals_result_xgb['train'][par])):
            np.testing.assert_allclose(evals_result_xgb['train'][par][i], evals_result_mxgb['train'][par][i], atol=0.011)
    predictions = bst.predict(xgb_dmatrix)
    modin_predictions = modin_bst.predict(mxgb_dmatrix)
    preds = pd.DataFrame(predictions).apply(round)
    modin_preds = modin_predictions.apply(round)
    val = accuracy_score(y, preds)
    modin_val = accuracy_score(modin_y, modin_preds)
    np.testing.assert_allclose(val, modin_val, atol=0.002, rtol=0.002)