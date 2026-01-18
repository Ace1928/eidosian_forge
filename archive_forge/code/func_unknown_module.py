import math
import re
import textwrap
import operator
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase
def unknown_module():
    return numpyz.int32(0)