import numpy
import pytest
from thinc.layers import premap_ids, remap_ids, remap_ids_v2
def test_remap(keys, mapper):
    remap = remap_ids(mapper, default=99)
    values, _ = remap(keys, False)
    numpy.testing.assert_equal(values.squeeze(), numpy.asarray(range(len(keys))))