from unittest import mock
import numpy as np
import pandas as pd
import pytest
from tensorflow import nest
import autokeras as ak
from autokeras import test_utils
@mock.patch('autokeras.AutoModel.fit')
@mock.patch('autokeras.AutoModel.evaluate')
def test_structured_clf_evaluate_call_automodel_evaluate(evaluate, fit, tmp_path):
    auto_model = ak.StructuredDataClassifier(directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=test_utils.TRAIN_CSV_PATH, y='survived')
    auto_model.evaluate(x=test_utils.TRAIN_CSV_PATH, y='survived')
    assert evaluate.is_called