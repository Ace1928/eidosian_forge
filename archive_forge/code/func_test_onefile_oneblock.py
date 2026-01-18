from __future__ import annotations
import os
import random
import pytest
import dask.bag as db
def test_onefile_oneblock(tmpdir):
    tmpdir = str(tmpdir)
    fn = os.path.join(tmpdir, 'one.avro')
    with open(fn, 'wb') as f:
        fastavro.writer(f, records=expected, schema=schema)
    b = db.read_avro(fn, blocksize=None)
    assert b.npartitions == 1
    assert b.compute() == expected