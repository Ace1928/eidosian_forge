import math
import numpy as np
import numbers
import re
import traceback
import multiprocessing as mp
import numba
from numba import njit, prange
from numba.core import config
from numba.tests.support import TestCase, tag, override_env_config
import unittest
def usecase_name(dtype, mode, vlen, name):
    """ Returns pretty name for given set of modes """
    return f'{dtype}_{mode}{vlen}_{name}'