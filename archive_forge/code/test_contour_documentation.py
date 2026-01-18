import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import requires_scipy
Test the fast-path option for filled contours.