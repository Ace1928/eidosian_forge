from unittest import SkipTest
from pyviz_comms import CommManager
from holoviews import Store, notebook_extension
from holoviews.core.options import OptionTree
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import bokeh, mpl
from holoviews.util import Options, OutputSettings, opts, output
from ..utils import LoggingComparisonTestCase
def test_opts_builder_repr_options_dotted(self):
    options = [Options('Bivariate.Test.Example', bandwidth=0.5, cmap='Blues'), Options('Points', size=2, logx=True)]
    expected = ["opts.Bivariate('Test.Example', bandwidth=0.5, cmap='Blues')", 'opts.Points(logx=True, size=2)']
    reprs = opts._builder_reprs(options)
    self.assertEqual(reprs, expected)