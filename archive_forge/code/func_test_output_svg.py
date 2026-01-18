from unittest import SkipTest
import holoviews as hv
from holoviews.core.options import Store
from pyviz_comms import CommManager
from holoviews.operation import Compositor
def test_output_svg(self):
    self.line_magic('output', "fig='svg'")
    self.assertEqual(hv.util.OutputSettings.options.get('fig', None), 'svg')