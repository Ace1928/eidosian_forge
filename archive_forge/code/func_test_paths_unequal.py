from holoviews import Bounds, Box, Contours, Ellipse, Path
from holoviews.element.comparison import ComparisonTestCase
def test_paths_unequal(self):
    try:
        self.assertEqual(self.path1, self.path2)
    except AssertionError as e:
        if not str(e).startswith('Path not almost equal to 6 decimals'):
            raise self.failureException('Path mismatch error not raised.')