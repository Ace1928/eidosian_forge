from unittest import TestCase, SkipTest
from hvplot.util import process_xarray  # noqa
def test_using_explicit_widgets_with_groupby_does_not_raise_error(self):
    import panel as pn
    x = pn.widgets.Select(name='x', value='sepal_length', options=self.cols)
    y = pn.widgets.Select(name='y', value='sepal_width', options=self.cols)
    pane = self.flowers.hvplot(x, y, groupby='species')
    assert isinstance(pane, pn.param.ParamFunction)