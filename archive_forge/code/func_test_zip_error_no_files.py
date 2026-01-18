import os
from pathlib import Path
import tarfile
import zipfile
import pytest
from pandas import DataFrame
import pandas._testing as tm
def test_zip_error_no_files(parser_and_data):
    parser, _, _ = parser_and_data
    with tm.ensure_clean() as path:
        with zipfile.ZipFile(path, mode='w'):
            pass
        with pytest.raises(ValueError, match='Zero files'):
            parser.read_csv(path, compression='zip')