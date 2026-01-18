import numpy as np
from holoviews.element import HSV, RGB, Curve, Image, QuadMesh, Raster
from holoviews.element.comparison import ComparisonTestCase
def test_quadmesh_to_trimesh(self):
    qmesh = QuadMesh(([0, 1], [0, 1], np.array([[0, 1], [2, 3]])))
    trimesh = qmesh.trimesh()
    simplices = np.array([[0, 1, 3, 0], [1, 2, 4, 2], [3, 4, 6, 1], [4, 5, 7, 3], [4, 3, 1, 0], [5, 4, 2, 2], [7, 6, 4, 1], [8, 7, 5, 3]])
    vertices = np.array([(-0.5, -0.5), (-0.5, 0.5), (-0.5, 1.5), (0.5, -0.5), (0.5, 0.5), (0.5, 1.5), (1.5, -0.5), (1.5, 0.5), (1.5, 1.5)])
    self.assertEqual(trimesh.array(), simplices)
    self.assertEqual(trimesh.nodes.array([0, 1]), vertices)