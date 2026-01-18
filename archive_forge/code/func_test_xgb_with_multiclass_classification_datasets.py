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
@pytest.mark.parametrize('data', [(load_iris(), {'num_class': 3}), (load_digits(), {'num_class': 10}), (load_wine(), {'num_class': 3})], ids=['load_iris', 'load_digits', 'load_wine'])
def test_xgb_with_multiclass_classification_datasets(data, num_actors, modin_type_y):
    dataset, param_ = data
    num_round = 10
    part_param = {'objective': 'multi:softprob', 'eval_metric': 'mlogloss'}
    param = {**param_, **part_param}
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
    assert len(evals_result_xgb['train']['mlogloss']) == len(evals_result_mxgb['train']['mlogloss'])
    for i in range(len(evals_result_xgb['train']['mlogloss'])):
        np.testing.assert_allclose(evals_result_xgb['train']['mlogloss'][i], evals_result_mxgb['train']['mlogloss'][i], atol=0.009)
    predictions = bst.predict(xgb_dmatrix)
    modin_predictions = modin_bst.predict(mxgb_dmatrix)
    array_preds = np.asarray([np.argmax(line) for line in predictions])
    modin_array_preds = np.asarray([np.argmax(line) for line in modin_predictions.to_numpy()])
    val = accuracy_score(y, array_preds)
    modin_val = accuracy_score(modin_y, modin_array_preds)
    np.testing.assert_allclose(val, modin_val)