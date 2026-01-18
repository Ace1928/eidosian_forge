from unittest import mock
import numpy as np
import pandas as pd
import pytest
from tensorflow import nest
import autokeras as ak
from autokeras import test_utils
def test_structured_data_col_type_no_name_error(tmp_path):
    with pytest.raises(ValueError) as info:
        clf = ak.StructuredDataClassifier(column_types={'age': 'numerical', 'parch': 'categorical'}, directory=tmp_path, seed=test_utils.SEED)
        clf.fit(x=np.random.rand(100, 30), y=np.random.rand(100, 1))
    assert 'column_names must be specified' in str(info.value)