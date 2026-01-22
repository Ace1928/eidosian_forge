import collections
import functools
import math
import multiprocessing
import os
import random
import subprocess
import sys
import threading
import itertools
from textwrap import dedent
import numpy as np
import unittest
import numba
from numba import jit, _helperlib, njit
from numba.core import types
from numba.tests.support import TestCase, compile_function, tag
from numba.core.errors import TypingError
def _accumulate_array_results(self, func, nresults):
    """
        Accumulate array results produced by *func* until they reach
        *nresults* elements.
        """
    res = []
    while len(res) < nresults:
        res += list(func().flat)
    return res[:nresults]