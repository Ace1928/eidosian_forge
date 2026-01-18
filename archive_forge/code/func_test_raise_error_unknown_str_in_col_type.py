from unittest import mock
import numpy as np
import pandas as pd
import pytest
from tensorflow import nest
import autokeras as ak
from autokeras import test_utils
def test_raise_error_unknown_str_in_col_type(tmp_path):
    with pytest.raises(ValueError) as info:
        ak.StructuredDataClassifier(column_types={'age': 'num', 'parch': 'categorical'}, directory=tmp_path, seed=test_utils.SEED)
    assert 'column_types should be either "categorical"' in str(info.value)