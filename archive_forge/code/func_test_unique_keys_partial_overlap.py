from holoviews import Curve, DynamicMap, HoloMap
from holoviews.core.traversal import unique_dimkeys
from holoviews.element.comparison import ComparisonTestCase
def test_unique_keys_partial_overlap(self):
    hmap1 = HoloMap({(i, j): Curve(range(10)) for i in range(5) for j in range(3)}, kdims=['A', 'B'])
    hmap2 = HoloMap({i: Curve(range(10)) for i in range(5)}, kdims=['A'])
    dims, keys = unique_dimkeys(hmap1 + hmap2)
    self.assertEqual(hmap1.kdims, dims)
    self.assertEqual(keys, [(i, j) for i in range(5) for j in list(range(3))])