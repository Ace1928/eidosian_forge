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
def validate_select_k(select_k_indices, tbl, sort_keys, stable_sort=False):
    sorted_indices = pc.sort_indices(tbl, sort_keys=sort_keys)
    head_k_indices = sorted_indices.slice(0, len(select_k_indices))
    if stable_sort:
        assert select_k_indices == head_k_indices
    else:
        expected = pc.take(tbl, head_k_indices)
        actual = pc.take(tbl, select_k_indices)
        assert actual == expected