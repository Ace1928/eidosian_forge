import time
import hashlib
import sys
import gc
import io
import collections
import itertools
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from joblib.hashing import hash
from joblib.func_inspect import filter_args
from joblib.memory import Memory
from joblib.testing import raises, skipif, fixture, parametrize
from joblib.test.common import np, with_numpy
@fixture(scope='function')
@with_numpy
def three_np_arrays():
    rnd = np.random.RandomState(0)
    arr1 = rnd.random_sample((10, 10))
    arr2 = arr1.copy()
    arr3 = arr2.copy()
    arr3[0] += 1
    return (arr1, arr2, arr3)