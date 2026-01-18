import numpy as np
import cartopy.crs as ccrs
from geoviews.element import Image, VectorField, WindBarbs
from geoviews.element.comparison import ComparisonTestCase
from geoviews.operation import project
def test_project_vectorfield(self):
    xs = np.linspace(10, 50, 2)
    X, Y = np.meshgrid(xs, xs)
    U, V = (5 * X, 1 * Y)
    A = np.arctan2(V, U)
    M = np.hypot(U, V)
    crs = ccrs.PlateCarree()
    vectorfield = VectorField((X, Y, A, M), crs=crs)
    projection = ccrs.Orthographic()
    projected = project(vectorfield, projection=projection)
    assert projected.crs == projection
    xs, ys, ang, ms = (vectorfield.dimension_values(i) for i in range(4))
    us = np.sin(ang) * -ms
    vs = np.cos(ang) * -ms
    u, v = projection.transform_vectors(crs, xs, ys, us, vs)
    a, m = (np.arctan2(v, u).T, np.hypot(u, v).T)
    np.testing.assert_allclose(projected.dimension_values('Angle'), a.flatten())
    np.testing.assert_allclose(projected.dimension_values('Magnitude'), m.flatten())