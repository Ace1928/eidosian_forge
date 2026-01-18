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
def test_variance():
    data = [1, 2, 3, 4, 5, 6, 7, 8]
    assert pc.variance(data).as_py() == 5.25
    assert pc.variance(data, ddof=0).as_py() == 5.25
    assert pc.variance(data, ddof=1).as_py() == 6.0