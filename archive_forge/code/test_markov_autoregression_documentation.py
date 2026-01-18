import warnings
import os
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pandas as pd
import pytest
from statsmodels.tools import add_constant
from statsmodels.tsa.regime_switching import markov_autoregression

Tests for Markov Autoregression models

Author: Chad Fulton
License: BSD-3
