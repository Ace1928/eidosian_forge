import unittest
import math
import sys
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, tag
def usecase_int64_neg():
    return -9223372036854775808