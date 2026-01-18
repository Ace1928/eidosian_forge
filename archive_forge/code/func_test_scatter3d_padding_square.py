from holoviews.element import Scatter3D
from .test_plot import TestMPLPlot, mpl_renderer
def test_scatter3d_padding_square(self):
    scatter3d = Scatter3D([(0, 1, 2), (1, 2, 3), (2, 3, 4)]).opts(padding=0.1)
    plot = mpl_renderer.get_plot(scatter3d)
    x_range, y_range = (plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim())
    z_range = plot.handles['axis'].get_zlim()
    self.assertEqual(x_range[0], -0.2)
    self.assertEqual(x_range[1], 2.2)
    self.assertEqual(y_range[0], 0.8)
    self.assertEqual(y_range[1], 3.2)
    self.assertEqual(z_range[0], 1.8)
    self.assertEqual(z_range[1], 4.2)