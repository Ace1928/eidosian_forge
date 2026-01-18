from os import path
import pickle
import numpy as np
def test_numpy_load_numpy_2_0_file(self):
    """
        Checks that `numpy.load` for NumPy 1.26 is able to load pickles
        created with NumPy 2.0 without errors/warnings.
        """
    arr = np.load(self.filename, encoding='latin1', allow_pickle=True)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (73,) and arr.dtype == np.float64