import pytest
import numpy as np
from numpy.linalg import norm as npnorm
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import scipy.sparse
from scipy.sparse.linalg import norm as spnorm
Test functions for the sparse.linalg.norm module
