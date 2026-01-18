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
def test_partition_nth_null_placement():
    data = list(range(10)) + [None] * 10
    random.shuffle(data)
    for pivot in (0, 7, 13, 19):
        for null_placement in ('at_start', 'at_end'):
            indices = pc.partition_nth_indices(data, pivot=pivot, null_placement=null_placement)
            check_partition_nth(data, indices, pivot, null_placement)