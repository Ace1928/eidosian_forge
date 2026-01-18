from unittest import mock
import numpy as np
import pandas as pd
import pytest
from tensorflow import nest
import autokeras as ak
from autokeras import test_utils
@mock.patch('autokeras.AutoModel.fit')
@mock.patch('autokeras.AutoModel.predict')
def test_structured_clf_predict_csv_call_automodel_predict(predict, fit, tmp_path):
    auto_model = ak.StructuredDataClassifier(directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=test_utils.TRAIN_CSV_PATH, y='survived')
    auto_model.predict(x=test_utils.TEST_CSV_PATH)
    assert predict.is_called