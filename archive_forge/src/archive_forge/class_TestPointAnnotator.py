from holoviews import Overlay
from holoviews.annotators import PathAnnotator, PointAnnotator, annotate
from holoviews.element import Path, Points, Table
from holoviews.element.tiles import EsriStreet, Tiles
from holoviews.tests.plotting.bokeh.test_plot import TestBokehPlot
class TestPointAnnotator(TestBokehPlot):

    def test_add_annotations(self):
        annotator = PointAnnotator(Points([]), annotations=['Label'])
        self.assertIn('Label', annotator.object)

    def test_add_name(self):
        annotator = PointAnnotator(name='Test Annotator', annotations=['Label'])
        self.assertEqual(annotator._stream.tooltip, 'Test Annotator Tool')
        self.assertEqual(annotator._table.label, 'Test Annotator')
        self.assertEqual(annotator.editor._names, ['Test Annotator'])

    def test_annotation_type(self):
        annotator = PointAnnotator(Points([(1, 2)]), annotations={'Int': int})
        expected = Table([(1, 2, 0)], ['x', 'y'], vdims=['Int'], label='PointAnnotator')
        self.assertEqual(annotator._table, expected)

    def test_replace_object(self):
        annotator = PointAnnotator(Points([]), annotations=['Label'])
        annotator.object = Points([(1, 2)])
        self.assertIn('Label', annotator.object)
        expected = Table([(1, 2, '')], ['x', 'y'], vdims=['Label'], label='PointAnnotator')
        self.assertEqual(annotator._table, expected)
        self.assertIs(annotator._link.target, annotator._table)

    def test_stream_update(self):
        annotator = PointAnnotator(Points([(1, 2)]), annotations=['Label'])
        annotator._stream.event(data={'x': [1], 'y': [2], 'Label': ['A']})
        self.assertEqual(annotator.object, Points([(1, 2, 'A')], vdims=['Label']))