from unittest import SkipTest
import holoviews as hv
from holoviews.core.options import Store
from pyviz_comms import CommManager
from holoviews.operation import Compositor
def test_output_widgets_live(self):
    self.line_magic('output', "widgets='live'")
    self.assertEqual(hv.util.OutputSettings.options.get('widgets', None), 'live')