import json
import os
import re
from typing import TYPE_CHECKING
import fsspec
import numpy as np
import pandas
import pandas._libs.lib as lib
from fsspec.core import url_to_fs
from fsspec.spec import AbstractBufferedFile
from packaging import version
from pandas.io.common import stringify_path
from modin.config import MinPartitionSize, NPartitions
from modin.core.io.column_stores.column_store_dispatcher import ColumnStoreDispatcher
from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings
def _get_fastparquet_files(self):
    if '*' in self.path:
        files = self.fs.glob(self.path)
    elif self.fs.isfile(self.path):
        files = self.fs.find(self.path)
    else:
        files = [f for f in self.fs.find(self.path) if f.endswith('.parquet') or f.endswith('.parq')]
    return files