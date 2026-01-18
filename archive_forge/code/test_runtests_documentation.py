import os
import sys
import subprocess
from numba import cuda
import unittest
import itertools
returns a filter on strings starting with 'numba.', useful for
        selecting the 'numba' test names from a test listing.