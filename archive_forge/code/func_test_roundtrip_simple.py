from __future__ import annotations
import os
import random
import pytest
import dask.bag as db
def test_roundtrip_simple(tmpdir):
    from dask.delayed import Delayed
    tmpdir = str(tmpdir)
    fn = os.path.join(tmpdir, 'out*.avro')
    b = db.from_sequence([{'a': i} for i in [1, 2, 3, 4, 5]], npartitions=2)
    schema = {'name': 'Test', 'type': 'record', 'fields': [{'name': 'a', 'type': 'int'}]}
    out = b.to_avro(fn, schema, compute=False)
    assert isinstance(out[0], Delayed)
    out = b.to_avro(fn, schema)
    assert len(out) == 2
    b2 = db.read_avro(fn)
    assert b.compute() == b2.compute()