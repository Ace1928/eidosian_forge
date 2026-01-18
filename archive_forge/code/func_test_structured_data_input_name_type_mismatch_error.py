from unittest import mock
import numpy as np
import pandas as pd
import pytest
from tensorflow import nest
import autokeras as ak
from autokeras import test_utils
def test_structured_data_input_name_type_mismatch_error(tmp_path):
    with pytest.raises(ValueError) as info:
        clf = ak.StructuredDataClassifier(column_types={'_age': 'numerical', 'parch': 'categorical'}, column_names=['age', 'fare'], directory=tmp_path, seed=test_utils.SEED)
        clf.fit(x=test_utils.TRAIN_CSV_PATH, y='survived')
    assert 'column_names and column_types are mismatched.' in str(info.value)