import os
import sys
import subprocess
from numba import cuda
import unittest
import itertools
def _get_numba_tests_from_listing(self, listing):
    """returns a filter on strings starting with 'numba.', useful for
        selecting the 'numba' test names from a test listing."""
    return filter(lambda x: x.startswith('numba.'), listing)