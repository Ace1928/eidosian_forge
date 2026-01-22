import pickle
from multiprocessing import Pool
import affine
def _mp_proc(x):
    assert isinstance(x, affine.Affine)
    return x