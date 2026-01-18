import io
import os
import sys
from zipfile import ZipFile
from _csv import Error
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('compression', ['zip', 'infer'])
@pytest.mark.parametrize('archive_name', ['test_to_csv.csv', 'test_to_csv.zip'])
def test_to_csv_zip_arguments(self, compression, archive_name):
    df = DataFrame({'ABC': [1]})
    with tm.ensure_clean('to_csv_archive_name.zip') as path:
        df.to_csv(path, compression={'method': compression, 'archive_name': archive_name})
        with ZipFile(path) as zp:
            assert len(zp.filelist) == 1
            archived_file = zp.filelist[0].filename
            assert archived_file == archive_name