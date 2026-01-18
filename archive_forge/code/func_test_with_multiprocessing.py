import pickle
from multiprocessing import Pool
import affine
def test_with_multiprocessing():
    a1 = affine.Affine(1, 2, 3, 4, 5, 6)
    a2 = affine.Affine(6, 5, 4, 3, 2, 1)
    results = Pool(2).map(_mp_proc, [a1, a2])
    for expected, actual in zip([a1, a2], results):
        assert expected == actual