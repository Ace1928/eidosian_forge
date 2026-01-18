import numpy
import pytest
from thinc.layers import premap_ids, remap_ids, remap_ids_v2
def test_remap_premap_eq(keys, mapper):
    remap = remap_ids(mapper, default=99)
    remap_v2 = remap_ids_v2(mapper, default=99)
    premap = premap_ids(mapper, default=99)
    values1, _ = remap(keys, False)
    values2, _ = remap_v2(keys, False)
    values3, _ = premap(keys, False)
    numpy.testing.assert_equal(values1, values2)
    numpy.testing.assert_equal(values2, values3)