from unittest import mock
import numpy as np
import pandas as pd
import pytest
from tensorflow import nest
import autokeras as ak
from autokeras import test_utils
@mock.patch('autokeras.AutoModel.fit')
def test_structured_data_clf_convert_csv_to_df_and_np(fit, tmp_path):
    auto_model = ak.StructuredDataClassifier(directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=test_utils.TRAIN_CSV_PATH, y='survived', epochs=2, validation_data=(test_utils.TEST_CSV_PATH, 'survived'))
    _, kwargs = fit.call_args_list[0]
    assert isinstance(kwargs['x'], pd.DataFrame)
    assert isinstance(kwargs['y'], np.ndarray)