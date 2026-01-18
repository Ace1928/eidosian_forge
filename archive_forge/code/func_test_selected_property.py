from holoviews import Overlay
from holoviews.annotators import PathAnnotator, PointAnnotator, annotate
from holoviews.element import Path, Points, Table
from holoviews.element.tiles import EsriStreet, Tiles
from holoviews.tests.plotting.bokeh.test_plot import TestBokehPlot
def test_selected_property(self):
    annotator = annotate.instance()
    annotator(Points([(1, 2), (2, 3)]), annotations=['Label'])
    annotator.annotator._selection.update(index=[1])
    self.assertEqual(annotator.selected, Points([(2, 3, '')], vdims='Label'))