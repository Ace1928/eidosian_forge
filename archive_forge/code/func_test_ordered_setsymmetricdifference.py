import copy
import itertools
import logging
import pickle
from io import StringIO
from collections import namedtuple as NamedTuple
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import native_numeric_types, native_types
import pyomo.core.base.set as SetModule
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.initializer import (
from pyomo.core.base.set import (
from pyomo.environ import (
def test_ordered_setsymmetricdifference(self):
    self._verify_ordered_symdifference(SetOf([3, 2, 1, 5, 4]), SetOf([0, 1, 4]))
    self._verify_ordered_symdifference(SetOf([3, 2, 1, 5, 4]), [0, 1, 4])
    self._verify_ordered_symdifference([3, 2, 1, 5, 4], SetOf([0, 1, 4]))