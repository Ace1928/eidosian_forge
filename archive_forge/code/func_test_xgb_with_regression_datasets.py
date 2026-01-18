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
@pytest.mark.parametrize('data', [(load_diabetes(), {'eta': 0.01})], ids=['load_diabetes'])
def test_xgb_with_regression_datasets(data, num_actors, modin_type_y):
    dataset, param = data
    num_round = 10
    X_df = pd.DataFrame(dataset.data)
    y_df = modin_type_y(dataset.target)
    X_train, X_test = train_test_split(X_df)
    y_train, y_test = train_test_split(y_df)
    train_xgb_dmatrix = xgboost.DMatrix(X_train, label=y_train)
    test_xgb_dmatrix = xgboost.DMatrix(X_test, label=y_test)
    train_mxgb_dmatrix = xgb.DMatrix(X_train, label=y_train)
    test_mxgb_dmatrix = xgb.DMatrix(X_test, label=y_test)
    evals_result_xgb = {}
    evals_result_mxgb = {}
    verbose_eval = False
    bst = xgboost.train(param, train_xgb_dmatrix, num_round, evals_result=evals_result_xgb, evals=[(train_xgb_dmatrix, 'train'), (test_xgb_dmatrix, 'test')], verbose_eval=verbose_eval)
    modin_bst = xgb.train(param, train_mxgb_dmatrix, num_round, evals_result=evals_result_mxgb, evals=[(train_mxgb_dmatrix, 'train'), (test_mxgb_dmatrix, 'test')], num_actors=num_actors, verbose_eval=verbose_eval)
    for param in ['train', 'test']:
        assert len(evals_result_xgb[param]['rmse']) == len(evals_result_mxgb[param]['rmse'])
        for i in range(len(evals_result_xgb[param]['rmse'])):
            np.testing.assert_allclose(evals_result_xgb[param]['rmse'][i], evals_result_mxgb[param]['rmse'][i], rtol=0.0007)
    predictions = bst.predict(train_xgb_dmatrix)
    modin_predictions = modin_bst.predict(train_mxgb_dmatrix)
    val = mean_squared_error(y_train, predictions)
    modin_val = mean_squared_error(y_train, modin_predictions)
    np.testing.assert_allclose(val, modin_val, rtol=1.25e-05)