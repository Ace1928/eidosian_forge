from unittest import skipIf
import param
from holoviews.core import DynamicMap, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PlotSize, RangeXY, Stream
def test_traverse_derived_streams(self):
    from holoviews.tests.test_streams import Val
    decollated = self.dmap_derived.decollate()
    self.assertIsInstance(decollated, DynamicMap)
    self.assertEqual(len(decollated.streams), 3)
    for stream in decollated.streams:
        self.assertIsInstance(stream, Val)
    expected = self.dmap_derived.callback.callable(6.0)
    decollated.streams[0].event(v=1.0)
    decollated.streams[1].event(v=2.0)
    decollated.streams[2].event(v=3.0)
    result = decollated[()]
    self.assertEqual(expected, result)