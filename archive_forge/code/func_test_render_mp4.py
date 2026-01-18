import subprocess
from unittest import SkipTest
import numpy as np
import panel as pn
import param
from matplotlib import style
from panel.widgets import DiscreteSlider, FloatSlider, Player
from pyviz_comms import CommManager
from holoviews import Curve, DynamicMap, GridSpace, HoloMap, Image, ItemTable, Table
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.mpl import CurvePlot, MPLRenderer
from holoviews.plotting.renderer import Renderer
from holoviews.streams import Stream
def test_render_mp4(self):
    devnull = subprocess.DEVNULL
    try:
        subprocess.call(['ffmpeg', '-h'], stdout=devnull, stderr=devnull)
    except Exception:
        raise SkipTest('ffmpeg not available, skipping mp4 export test')
    data, metadata = self.renderer.components(self.map1, 'mp4')
    self.assertIn("<source src='data:video/mp4", data['text/html'])