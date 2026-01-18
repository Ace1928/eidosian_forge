import unittest
import numpy as np
import scipy.linalg
from skimage import data, img_as_float
from pygsp import graphs
def test_sensor(self):
    graphs.Sensor(regular=True)
    graphs.Sensor(regular=False)
    graphs.Sensor(distribute=True)
    graphs.Sensor(distribute=False)
    graphs.Sensor(connected=True)
    graphs.Sensor(connected=False)