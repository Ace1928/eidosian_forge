import numpy as np
import skimage.graph.mcp as mcp
from skimage._shared.testing import assert_array_equal, assert_almost_equal, parametrize
from skimage._shared._warnings import expected_warnings
def _test_random(shape):
    a = np.random.rand(*shape).astype(np.float32)
    starts = [[0] * len(shape), [-1] * len(shape), (np.random.rand(len(shape)) * shape).astype(int)]
    ends = [(np.random.rand(len(shape)) * shape).astype(int) for i in range(4)]
    with expected_warnings(['Upgrading NumPy' + warning_optional]):
        m = mcp.MCP(a, fully_connected=True)
    costs, offsets = m.find_costs(starts)
    for point in [(np.random.rand(len(shape)) * shape).astype(int) for i in range(4)]:
        m.traceback(point)
    m._reset()
    m.find_costs(starts, ends)
    for end in ends:
        m.traceback(end)
    return (a, costs, offsets)