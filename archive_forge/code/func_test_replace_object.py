from holoviews import Overlay
from holoviews.annotators import PathAnnotator, PointAnnotator, annotate
from holoviews.element import Path, Points, Table
from holoviews.element.tiles import EsriStreet, Tiles
from holoviews.tests.plotting.bokeh.test_plot import TestBokehPlot
def test_replace_object(self):
    annotator = PathAnnotator(Path([]), annotations=['Label'], vertex_annotations=['Value'])
    annotator.object = Path([(1, 2), (2, 3), (0, 0)])
    self.assertIn('Label', annotator.object)
    expected = Table([''], kdims=['Label'], label='PathAnnotator')
    self.assertEqual(annotator._table, expected)
    expected = Table([], ['x', 'y'], 'Value', label='PathAnnotator Vertices')
    self.assertEqual(annotator._vertex_table, expected)
    self.assertIs(annotator._link.target, annotator._table)
    self.assertIs(annotator._vertex_link.target, annotator._vertex_table)