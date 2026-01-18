import panel as pn
import param
from panel.widgets import DiscreteSlider, FloatSlider, Player
from pyviz_comms import CommManager
from holoviews import Curve, DynamicMap, HoloMap, Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.plotly import PlotlyRenderer
from holoviews.plotting.renderer import Renderer
from holoviews.streams import Stream
def test_render_static(self):
    curve = Curve([])
    obj, _ = self.renderer._validate(curve, None)
    self.assertIsInstance(obj, pn.pane.HoloViews)
    self.assertEqual(obj.center, True)
    self.assertIs(obj.renderer, self.renderer)
    self.assertEqual(obj.backend, 'plotly')