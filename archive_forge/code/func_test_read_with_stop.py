import pytest
import pandas as pd
import pandas._testing as tm
def test_read_with_stop(self, pytables_hdf5_file):
    path, objname, df = pytables_hdf5_file
    result = pd.read_hdf(path, key=objname, stop=1)
    expected = df[:1].reset_index(drop=True)
    tm.assert_frame_equal(result, expected, check_index_type=True)