from unittest import SkipTest
from pyviz_comms import CommManager
from holoviews import Store, notebook_extension
from holoviews.core.options import OptionTree
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import bokeh, mpl
from holoviews.util import Options, OutputSettings, opts, output
from ..utils import LoggingComparisonTestCase
class TestOptsUtil(LoggingComparisonTestCase):
    """
    Mirrors the magic tests in TestOptsMagic
    """

    def setUp(self):
        self.backend = Store.current_backend
        Store.current_backend = 'matplotlib'
        self.store_copy = OptionTree(sorted(Store.options().items()), groups=Options._option_groups)
        super().setUp()

    def tearDown(self):
        Store.current_backend = self.backend
        Store.options(val=self.store_copy)
        Store._custom_options = {k: {} for k in Store._custom_options.keys()}
        super().tearDown()

    def test_opts_builder_repr(self):
        magic = "Bivariate [bandwidth=0.5] (cmap='jet') Points [logx=True] (size=2)"
        expected = ["opts.Bivariate(bandwidth=0.5, cmap='jet')", 'opts.Points(logx=True, size=2)']
        reprs = opts._builder_reprs(magic)
        self.assertEqual(reprs, expected)

    def test_opts_builder_repr_line_magic(self):
        magic = "%opts Bivariate [bandwidth=0.5] (cmap='jet') Points [logx=True] (size=2)"
        expected = ["opts.Bivariate(bandwidth=0.5, cmap='jet')", 'opts.Points(logx=True, size=2)']
        reprs = opts._builder_reprs(magic)
        self.assertEqual(reprs, expected)

    def test_opts_builder_repr_cell_magic(self):
        magic = "%%opts Bivariate [bandwidth=0.5] (cmap='jet') Points [logx=True] (size=2)"
        expected = ["opts.Bivariate(bandwidth=0.5, cmap='jet')", 'opts.Points(logx=True, size=2)']
        reprs = opts._builder_reprs(magic)
        self.assertEqual(reprs, expected)

    def test_opts_builder_repr_options_dotted(self):
        options = [Options('Bivariate.Test.Example', bandwidth=0.5, cmap='Blues'), Options('Points', size=2, logx=True)]
        expected = ["opts.Bivariate('Test.Example', bandwidth=0.5, cmap='Blues')", 'opts.Points(logx=True, size=2)']
        reprs = opts._builder_reprs(options)
        self.assertEqual(reprs, expected)