import datetime
from io import BytesIO
import re
import numpy as np
import pytest
from pandas import (
from pandas.tests.io.pytables.common import ensure_clean_store
from pandas.io.pytables import (
def test_unsuppored_hdf_file_error(datapath):
    data_path = datapath('io', 'data', 'legacy_hdf/incompatible_dataset.h5')
    message = 'Dataset\\(s\\) incompatible with Pandas data types, not table, or no datasets found in HDF5 file.'
    with pytest.raises(ValueError, match=message):
        read_hdf(data_path)