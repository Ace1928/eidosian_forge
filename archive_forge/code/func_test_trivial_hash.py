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
@parametrize('obj1', input_list)
@parametrize('obj2', input_list)
def test_trivial_hash(obj1, obj2):
    """Smoke test hash on various types."""
    are_hashes_equal = hash(obj1) == hash(obj2)
    are_objs_identical = obj1 is obj2
    assert are_hashes_equal == are_objs_identical