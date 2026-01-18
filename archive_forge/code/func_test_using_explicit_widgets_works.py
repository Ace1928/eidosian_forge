from unittest import TestCase, SkipTest
from hvplot.util import process_xarray  # noqa
def test_using_explicit_widgets_works(self):
    import panel as pn
    x = pn.widgets.Select(name='x', value='sepal_length', options=self.cols)
    y = pn.widgets.Select(name='y', value='sepal_width', options=self.cols)
    kind = pn.widgets.Select(name='kind', value='scatter', options=['bivariate', 'scatter'])
    by_species = pn.widgets.Checkbox(name='By species')
    color = pn.widgets.ColorPicker(value='#ff0000')

    @pn.depends(by_species.param.value, color.param.value)
    def by_species_fn(by_species, color):
        return 'species' if by_species else color
    self.flowers.hvplot(x, y=y, kind=kind.param.value, c=color)