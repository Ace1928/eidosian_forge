from unittest import SkipTest
import holoviews as hv
from holoviews.core.options import Store
from pyviz_comms import CommManager
from holoviews.operation import Compositor
class TestCompositorMagic(ExtensionTestCase):

    def setUp(self):
        super().setUp()
        self.cell('import numpy as np')
        self.cell('from holoviews.element import Image')
        self.definitions = list(Compositor.definitions)
        Compositor.definitions[:] = []

    def tearDown(self):
        Compositor.definitions[:] = self.definitions
        super().tearDown()

    def test_display_compositor_definition(self):
        definition = ' display factory(Image * Image * Image) RGBTEST'
        self.line_magic('compositor', definition)
        compositors = [c for c in Compositor.definitions if c.group == 'RGBTEST']
        self.assertEqual(len(compositors), 1)
        self.assertEqual(compositors[0].group, 'RGBTEST')
        self.assertEqual(compositors[0].mode, 'display')

    def test_data_compositor_definition(self):
        definition = ' data transform(Image * Image) HCSTEST'
        self.line_magic('compositor', definition)
        compositors = [c for c in Compositor.definitions if c.group == 'HCSTEST']
        self.assertEqual(len(compositors), 1)
        self.assertEqual(compositors[0].group, 'HCSTEST')
        self.assertEqual(compositors[0].mode, 'data')