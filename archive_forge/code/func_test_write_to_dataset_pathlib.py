import datetime
import inspect
import os
import pathlib
import numpy as np
import pytest
import unittest.mock as mock
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_write_to_dataset_pathlib(tempdir):
    _test_write_to_dataset_with_partitions(tempdir / 'test1')
    _test_write_to_dataset_no_partitions(tempdir / 'test2')