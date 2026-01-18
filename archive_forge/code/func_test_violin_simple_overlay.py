import numpy as np
from holoviews.element import Violin
from .test_plot import TestMPLPlot, mpl_renderer
def test_violin_simple_overlay(self):
    values = np.random.rand(100)
    violin = Violin(values) * Violin(values)
    plot = mpl_renderer.get_plot(violin)
    p1, p2 = plot.subplots.values()
    self.assertEqual(p1.handles['boxes'][0].get_path().vertices, p2.handles['boxes'][0].get_path().vertices)
    for b1, b2 in zip(p1.handles['bodies'][0].get_paths(), p2.handles['bodies'][0].get_paths()):
        self.assertEqual(b1.vertices, b2.vertices)