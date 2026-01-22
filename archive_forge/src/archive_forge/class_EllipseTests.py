import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
class EllipseTests(ComparisonTestCase):

    def setUp(self):
        self.pentagon = np.array([[0.0, 0.5], [0.475528258, 0.154508497], [0.293892626, -0.404508497], [-0.293892626, -0.404508497], [-0.475528258, 0.154508497], [-1.2246468e-16, 0.5]])
        self.squashed = np.array([[0.0, 1.0], [0.475528258, 0.309016994], [0.293892626, -0.809016994], [-0.293892626, -0.809016994], [-0.475528258, 0.309016994], [-1.2246468e-16, 1.0]])

    def test_ellipse_simple_constructor(self):
        ellipse = Ellipse(0, 0, 1, samples=100)
        self.assertEqual(len(ellipse.data[0]), 100)

    def test_ellipse_simple_constructor_pentagon(self):
        ellipse = Ellipse(0, 0, 1, samples=6)
        self.assertEqual(np.allclose(ellipse.data[0], self.pentagon), True)

    def test_ellipse_tuple_constructor_squashed(self):
        ellipse = Ellipse(0, 0, (1, 2), samples=6)
        self.assertEqual(np.allclose(ellipse.data[0], self.squashed), True)

    def test_ellipse_simple_constructor_squashed_aspect(self):
        ellipse = Ellipse(0, 0, 2, aspect=0.5, samples=6)
        self.assertEqual(np.allclose(ellipse.data[0], self.squashed), True)