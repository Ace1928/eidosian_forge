import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from scipy.linalg import cho_solve_banded
from statsmodels import datasets
from statsmodels.tsa.statespace import (sarimax, structural, dynamic_factor,

Tests for CFA simulation smoothing

Author: Chad Fulton
License: BSD-3
