import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
Testing `dict()` and `{}` usage that are redirected to
    `numba.typed.Dict`.
    