import unittest
import numpy as np
import scipy.linalg
from skimage import data, img_as_float
from pygsp import graphs
def test_swissroll(self):
    graphs.SwissRoll(srtype='uniform')
    graphs.SwissRoll(srtype='classic')
    graphs.SwissRoll(noise=True)
    graphs.SwissRoll(noise=False)
    graphs.SwissRoll(dim=2)
    graphs.SwissRoll(dim=3)