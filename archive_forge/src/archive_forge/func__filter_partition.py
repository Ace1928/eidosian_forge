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
def _filter_partition(df, part_keys):
    predicate = np.ones(len(df), dtype=bool)
    to_drop = []
    for name, value in part_keys:
        to_drop.append(name)
        if isinstance(value, (datetime.date, datetime.datetime)):
            value = pd.Timestamp(value)
        predicate &= df[name] == value
    return df[predicate].drop(to_drop, axis=1)