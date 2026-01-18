import unittest
import math
import sys
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, tag
def usecase_int64_func():
    return max(9223372036854775807, -9223372036854775808) + min(9223372036854775807, -9223372036854775808)