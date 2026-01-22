import os
import sys
import itertools
import traceback
import textwrap
import subprocess
import pytest
import numpy as np
from numpy import array, single, double, csingle, cdouble, dot, identity, matmul
from numpy.core import swapaxes
from numpy import multiply, atleast_2d, inf, asarray
from numpy import linalg
from numpy.linalg import matrix_power, norm, matrix_rank, multi_dot, LinAlgError
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
from numpy.testing import (
class LinalgTestCase:
    TEST_CASES = CASES

    def check_cases(self, require=set(), exclude=set()):
        """
        Run func on each of the cases with all of the tags in require, and none
        of the tags in exclude
        """
        for case in self.TEST_CASES:
            if case.tags & require != require:
                continue
            if case.tags & exclude:
                continue
            try:
                case.check(self.do)
            except Exception as e:
                msg = f'In test case: {case!r}\n\n'
                msg += traceback.format_exc()
                raise AssertionError(msg) from e