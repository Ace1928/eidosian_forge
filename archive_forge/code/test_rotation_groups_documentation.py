import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.constants import golden as phi
from scipy.spatial import cKDTree
Test that the elements of the rotation group are correctly
    mapped onto the identity rotation.