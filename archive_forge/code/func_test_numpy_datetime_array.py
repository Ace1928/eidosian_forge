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
@with_numpy
@parametrize('dtype', ['datetime64[s]', 'timedelta64[D]'])
def test_numpy_datetime_array(dtype):
    a_hash = hash(np.arange(10))
    array = np.arange(0, 10, dtype=dtype)
    assert hash(array) != a_hash