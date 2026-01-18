import unittest
import numpy as np
import scipy.linalg
from skimage import data, img_as_float
from pygsp import graphs
def test_set_coordinates(self):
    G = graphs.FullConnected()
    coords = self._rs.uniform(size=(G.N, 2))
    G.set_coordinates(coords)
    G.set_coordinates('ring2D')
    G.set_coordinates('random2D')
    G.set_coordinates('random3D')
    G.set_coordinates('spring')
    G.set_coordinates('spring', dim=3)
    G.set_coordinates('spring', dim=3, pos=G.coords)
    self.assertRaises(AttributeError, G.set_coordinates, 'community2D')
    G = graphs.Community()
    G.set_coordinates('community2D')
    self.assertRaises(ValueError, G.set_coordinates, 'invalid')