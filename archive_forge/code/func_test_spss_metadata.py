import datetime
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.filterwarnings('ignore::pandas.errors.ChainedAssignmentError')
@pytest.mark.filterwarnings('ignore:ChainedAssignmentError:FutureWarning')
def test_spss_metadata(datapath):
    fname = datapath('io', 'data', 'spss', 'labelled-num.sav')
    df = pd.read_spss(fname)
    metadata = {'column_names': ['VAR00002'], 'column_labels': [None], 'column_names_to_labels': {'VAR00002': None}, 'file_encoding': 'UTF-8', 'number_columns': 1, 'number_rows': 1, 'variable_value_labels': {'VAR00002': {1.0: 'This is one'}}, 'value_labels': {'labels0': {1.0: 'This is one'}}, 'variable_to_label': {'VAR00002': 'labels0'}, 'notes': [], 'original_variable_types': {'VAR00002': 'F8.0'}, 'readstat_variable_types': {'VAR00002': 'double'}, 'table_name': None, 'missing_ranges': {}, 'missing_user_values': {}, 'variable_storage_width': {'VAR00002': 8}, 'variable_display_width': {'VAR00002': 8}, 'variable_alignment': {'VAR00002': 'unknown'}, 'variable_measure': {'VAR00002': 'unknown'}, 'file_label': None, 'file_format': 'sav/zsav'}
    if Version(pyreadstat.__version__) >= Version('1.2.4'):
        metadata.update({'creation_time': datetime.datetime(2015, 2, 6, 14, 33, 36), 'modification_time': datetime.datetime(2015, 2, 6, 14, 33, 36)})
    assert df.attrs == metadata