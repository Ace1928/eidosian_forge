import numpy as np
from skimage.graph._graph import pixel_graph, central_pixel
def test_small_graph():
    g, n = pixel_graph(mask, connectivity=2)
    assert g.shape == (4, 4)
    assert len(g.data) == 8
    np.testing.assert_allclose(np.unique(g.data), [1, np.sqrt(2)])
    np.testing.assert_array_equal(n, [0, 4, 5, 7])