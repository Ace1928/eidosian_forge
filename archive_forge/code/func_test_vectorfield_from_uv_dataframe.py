import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_vectorfield_from_uv_dataframe(self):
    x = np.linspace(-1, 1, 4)
    X, Y = np.meshgrid(x, x)
    U, V = (5 * X, 5 * Y)
    df = pd.DataFrame({'x': X.flatten(), 'y': Y.flatten(), 'u': U.flatten(), 'v': V.flatten()})
    vectorfield = VectorField.from_uv(df, ['x', 'y'], ['u', 'v'])
    angle = np.arctan2(V, U)
    mag = np.hypot(U, V)
    kdims = [Dimension('x'), Dimension('y')]
    vdims = [Dimension('Angle', cyclic=True, range=(0, 2 * np.pi)), Dimension('Magnitude')]
    self.assertEqual(vectorfield.kdims, kdims)
    self.assertEqual(vectorfield.vdims, vdims)
    self.assertEqual(vectorfield.dimension_values(2, flat=False), angle.flat)
    self.assertEqual(vectorfield.dimension_values(3, flat=False), mag.flat)