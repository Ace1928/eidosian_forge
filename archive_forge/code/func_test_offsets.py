import numpy as np
import skimage.graph.mcp as mcp
from skimage._shared.testing import assert_array_equal, assert_almost_equal, parametrize
from skimage._shared._warnings import expected_warnings
def test_offsets():
    offsets = [(1, i) for i in range(10)] + [(1, -i) for i in range(1, 10)]
    with expected_warnings(['Upgrading NumPy' + warning_optional]):
        m = mcp.MCP(a, offsets=offsets)
    costs, traceback = m.find_costs([(1, 6)])
    assert_array_equal(traceback, [[-2, -2, -2, -2, -2, -2, -2, -2], [-2, -2, -2, -2, -2, -2, -1, -2], [15, 14, 13, 12, 11, 10, 0, 1], [10, 0, 1, 2, 3, 4, 5, 6], [10, 0, 1, 2, 3, 4, 5, 6], [10, 0, 1, 2, 3, 4, 5, 6], [10, 0, 1, 2, 3, 4, 5, 6], [10, 0, 1, 2, 3, 4, 5, 6]])
    assert hasattr(m, 'offsets')
    assert_array_equal(offsets, m.offsets)