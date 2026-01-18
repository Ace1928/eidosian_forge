from unittest import mock
import autokeras as ak
from autokeras import test_utils
@mock.patch('autokeras.AutoModel.fit')
@mock.patch('autokeras.AutoModel.evaluate')
def test_tsf_evaluate_call_automodel_evaluate(evaluate, fit, tmp_path):
    auto_model = ak.TimeseriesForecaster(lookback=10, directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=test_utils.TRAIN_CSV_PATH, y='survived')
    auto_model.evaluate(x=test_utils.TRAIN_CSV_PATH, y='survived')
    assert evaluate.is_called