from unittest import SkipTest
from pyviz_comms import CommManager
from holoviews import Store, notebook_extension
from holoviews.core.options import OptionTree
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import bokeh, mpl
from holoviews.util import Options, OutputSettings, opts, output
from ..utils import LoggingComparisonTestCase
class TestOutputUtil(ComparisonTestCase):

    def setUp(self):
        if notebook is None:
            raise SkipTest('Jupyter Notebook not available')
        notebook_extension(*BACKENDS)
        Store.current_backend = 'matplotlib'
        Store.renderers['matplotlib'] = mpl.MPLRenderer.instance()
        Store.renderers['bokeh'] = bokeh.BokehRenderer.instance()
        OutputSettings.options = dict(OutputSettings.defaults.items())
        super().setUp()

    def tearDown(self):
        Store.renderers['matplotlib'] = mpl.MPLRenderer.instance()
        Store.renderers['bokeh'] = bokeh.BokehRenderer.instance()
        OutputSettings.options = dict(OutputSettings.defaults.items())
        for renderer in Store.renderers.values():
            renderer.comm_manager = CommManager
        super().tearDown()

    def test_output_util_svg_string(self):
        self.assertEqual(OutputSettings.options.get('fig', None), None)
        output("fig='svg'")
        self.assertEqual(OutputSettings.options.get('fig', None), 'svg')

    def test_output_util_png_kwargs(self):
        self.assertEqual(OutputSettings.options.get('fig', None), None)
        output(fig='png')
        self.assertEqual(OutputSettings.options.get('fig', None), 'png')

    def test_output_util_backend_string(self):
        self.assertEqual(OutputSettings.options.get('backend', None), None)
        output("backend='bokeh'")
        self.assertEqual(OutputSettings.options.get('backend', None), 'bokeh')

    def test_output_util_backend_kwargs(self):
        self.assertEqual(OutputSettings.options.get('backend', None), None)
        output(backend='bokeh')
        self.assertEqual(OutputSettings.options.get('backend', None), 'bokeh')

    def test_output_util_object_noop(self):
        self.assertEqual(output("fig='svg'", 3), 3)