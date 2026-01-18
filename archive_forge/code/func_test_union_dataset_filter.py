import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
@pytest.mark.parametrize('dstype', ['fs', 'mem'])
def test_union_dataset_filter(tempdir, dstype):
    t1 = pa.table({'colA': [1, 2, 6, 8], 'col2': ['a', 'b', 'f', 'g']})
    t2 = pa.table({'colA': [9, 10, 11], 'col2': ['h', 'i', 'l']})
    if dstype == 'fs':
        ds.write_dataset(t1, tempdir / 't1', format='ipc')
        ds1 = ds.dataset(tempdir / 't1', format='ipc')
        ds.write_dataset(t2, tempdir / 't2', format='ipc')
        ds2 = ds.dataset(tempdir / 't2', format='ipc')
    elif dstype == 'mem':
        ds1 = ds.dataset(t1)
        ds2 = ds.dataset(t2)
    else:
        raise NotImplementedError
    filtered_union_ds = ds.dataset((ds1, ds2)).filter((pc.field('colA') < 3) | (pc.field('colA') == 9))
    assert filtered_union_ds.to_table() == pa.table({'colA': [1, 2, 9], 'col2': ['a', 'b', 'h']})
    joined = filtered_union_ds.join(ds.dataset(pa.table({'colB': [10, 20], 'col2': ['a', 'b']})), keys='col2', join_type='left outer')
    assert joined.to_table().sort_by('colA') == pa.table({'colA': [1, 2, 9], 'col2': ['a', 'b', 'h'], 'colB': [10, 20, None]})
    filtered_ds1 = ds1.filter(pc.field('colA') < 3)
    filtered_ds2 = ds2.filter(pc.field('colA') < 10)
    with pytest.raises(ValueError, match='currently not supported'):
        ds.dataset((filtered_ds1, filtered_ds2))