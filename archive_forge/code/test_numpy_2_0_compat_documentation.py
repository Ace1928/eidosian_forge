from os import path
import pickle
import numpy as np

        Checks that `numpy.load` for NumPy 1.26 is able to load pickles
        created with NumPy 2.0 without errors/warnings.
        