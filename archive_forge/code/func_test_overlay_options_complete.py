import numpy as np
from holoviews import Curve, HoloMap, Image, Overlay
from holoviews.core.options import Store, StoreOptions
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import mpl  # noqa Register backend
def test_overlay_options_complete(self):
    """
        Complete specification style.
        """
    data = [zip(range(10), range(10)), zip(range(5), range(5))]
    o = Overlay([Curve(c) for c in data]).opts({'Curve.Curve': {'show_grid': True, 'color': 'b'}})
    self.assertEqual(Store.lookup_options('matplotlib', o.Curve.I, 'plot').kwargs['show_grid'], True)
    self.assertEqual(Store.lookup_options('matplotlib', o.Curve.II, 'plot').kwargs['show_grid'], True)
    self.assertEqual(Store.lookup_options('matplotlib', o.Curve.I, 'style').kwargs['color'], 'b')
    self.assertEqual(Store.lookup_options('matplotlib', o.Curve.II, 'style').kwargs['color'], 'b')