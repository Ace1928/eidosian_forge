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
def _check_get_function(name, expected_func_cls, expected_ker_cls, min_num_kernels=1):
    func = pc.get_function(name)
    assert isinstance(func, expected_func_cls)
    n = func.num_kernels
    assert n >= min_num_kernels
    assert n == len(func.kernels)
    assert all((isinstance(ker, expected_ker_cls) for ker in func.kernels))