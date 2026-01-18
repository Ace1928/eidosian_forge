from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
def test_utf8_normalize():
    arr = pa.array(['01²3'])
    assert pc.utf8_normalize(arr, form='NFC') == arr
    assert pc.utf8_normalize(arr, form='NFKC') == pa.array(['0123'])
    assert pc.utf8_normalize(arr, 'NFD') == arr
    assert pc.utf8_normalize(arr, 'NFKD') == pa.array(['0123'])
    with pytest.raises(ValueError, match='"NFZ" is not a valid Unicode normalization form'):
        pc.utf8_normalize(arr, form='NFZ')