import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_vectorfield_from_uv(self):
    x = np.linspace(-1, 1, 4)
    X, Y = np.meshgrid(x, x)
    U, V = (3 * X, 4 * Y)
    vectorfield = VectorField.from_uv((X, Y, U, V))
    angle = np.arctan2(V, U)
    mag = np.hypot(U, V)
    kdims = [Dimension('x'), Dimension('y')]
    vdims = [Dimension('Angle', cyclic=True, range=(0, 2 * np.pi)), Dimension('Magnitude')]
    self.assertEqual(vectorfield.kdims, kdims)
    self.assertEqual(vectorfield.vdims, vdims)
    self.assertEqual(vectorfield.dimension_values(0), X.T.flatten())
    self.assertEqual(vectorfield.dimension_values(1), Y.T.flatten())
    self.assertEqual(vectorfield.dimension_values(2), angle.T.flatten())
    self.assertEqual(vectorfield.dimension_values(3), mag.T.flatten())