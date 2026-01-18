import os
import pathlib
import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import tobytes
from pyarrow.lib import ArrowInvalid, ArrowNotImplementedError
def test_serializing_multiple_expressions():
    schema = pa.schema([pa.field('x', pa.int32()), pa.field('y', pa.int32())])
    exprs = [pc.equal(pc.field('x'), 7), pc.equal(pc.field('x'), pc.field('y'))]
    buf = pa.substrait.serialize_expressions(exprs, ['first', 'second'], schema)
    returned = pa.substrait.deserialize_expressions(buf)
    assert schema == returned.schema
    assert len(returned.expressions) == 2
    norm_exprs = [pc.equal(pc.field(0), 7), pc.equal(pc.field(0), pc.field(1))]
    assert str(returned.expressions['first']) == str(norm_exprs[0])
    assert str(returned.expressions['second']) == str(norm_exprs[1])