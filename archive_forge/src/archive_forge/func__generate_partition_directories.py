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
def _generate_partition_directories(fs, base_dir, partition_spec, df):
    DEPTH = len(partition_spec)
    pathsep = getattr(fs, 'pathsep', getattr(fs, 'sep', '/'))

    def _visit_level(base_dir, level, part_keys):
        name, values = partition_spec[level]
        for value in values:
            this_part_keys = part_keys + [(name, value)]
            level_dir = pathsep.join([str(base_dir), '{}={}'.format(name, value)])
            fs.mkdir(level_dir)
            if level == DEPTH - 1:
                file_path = pathsep.join([level_dir, guid()])
                filtered_df = _filter_partition(df, this_part_keys)
                part_table = pa.Table.from_pandas(filtered_df)
                with fs.open(file_path, 'wb') as f:
                    _write_table(part_table, f)
                assert fs.exists(file_path)
                file_success = pathsep.join([level_dir, '_SUCCESS'])
                with fs.open(file_success, 'wb') as f:
                    pass
            else:
                _visit_level(level_dir, level + 1, this_part_keys)
                file_success = pathsep.join([level_dir, '_SUCCESS'])
                with fs.open(file_success, 'wb') as f:
                    pass
    _visit_level(base_dir, 0, [])