import panel as pn
import param
from panel.widgets import DiscreteSlider, FloatSlider, Player
from pyviz_comms import CommManager
from holoviews import Curve, DynamicMap, HoloMap, Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.plotly import PlotlyRenderer
from holoviews.plotting.renderer import Renderer
from holoviews.streams import Stream
def test_render_dynamicmap_with_stream(self):
    stream = Stream.define('Custom', y=2)()
    dmap = DynamicMap(lambda y: Curve([1, 2, y]), kdims=['y'], streams=[stream])
    obj, _ = self.renderer._validate(dmap, None)
    self.renderer.components(obj)
    [(plot, pane)] = obj._plots.values()
    y = plot.handles['fig']['data'][0]['y']
    self.assertEqual(y[2], 2)
    stream.event(y=3)
    y = plot.handles['fig']['data'][0]['y']
    self.assertEqual(y[2], 3)