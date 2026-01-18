from holoviews import Curve, DynamicMap, HoloMap
from holoviews.core.traversal import unique_dimkeys
from holoviews.element.comparison import ComparisonTestCase
def test_unique_keys_no_overlap_exception(self):
    hmap1 = HoloMap({i: Curve(range(10)) for i in range(5)}, kdims=['A'])
    hmap2 = HoloMap({i: Curve(range(10)) for i in range(3, 10)})
    exception = 'When combining HoloMaps into a composite plot their dimensions must be subsets of each other.'
    with self.assertRaisesRegex(Exception, exception):
        dims, keys = unique_dimkeys(hmap1 + hmap2)