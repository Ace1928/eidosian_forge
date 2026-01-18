import os
import pytest
import sys
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy.sparse.linalg._svdp import _svdp
from scipy.sparse import csr_matrix, csc_matrix
Check that the first k rows of u1 and u2 are orthogonal