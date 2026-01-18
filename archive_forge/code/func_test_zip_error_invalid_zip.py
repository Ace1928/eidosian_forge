import os
from pathlib import Path
import tarfile
import zipfile
import pytest
from pandas import DataFrame
import pandas._testing as tm
def test_zip_error_invalid_zip(parser_and_data):
    parser, _, _ = parser_and_data
    with tm.ensure_clean() as path:
        with open(path, 'rb') as f:
            with pytest.raises(zipfile.BadZipFile, match='File is not a zip file'):
                parser.read_csv(f, compression='zip')