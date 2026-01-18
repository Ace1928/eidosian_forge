from io import StringIO
import pytest
from pandas import read_sas
import pandas._testing as tm
def test_sas_archive(datapath):
    fname_uncompressed = datapath('io', 'sas', 'data', 'airline.sas7bdat')
    df_uncompressed = read_sas(fname_uncompressed)
    fname_compressed = datapath('io', 'sas', 'data', 'airline.sas7bdat.gz')
    df_compressed = read_sas(fname_compressed, format='sas7bdat')
    tm.assert_frame_equal(df_uncompressed, df_compressed)