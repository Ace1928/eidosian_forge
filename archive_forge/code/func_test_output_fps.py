from unittest import SkipTest
import holoviews as hv
from holoviews.core.options import Store
from pyviz_comms import CommManager
from holoviews.operation import Compositor
def test_output_fps(self):
    self.line_magic('output', 'fps=100')
    self.assertEqual(hv.util.OutputSettings.options.get('fps', None), 100)