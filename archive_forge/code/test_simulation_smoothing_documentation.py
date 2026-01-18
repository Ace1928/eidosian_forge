import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, structural, varmax
from statsmodels.tsa.statespace.simulation_smoother import (

    This is a very slow test to check that the distribution of simulated states
    (from the posterior) is correct in the presense of NaN values. Here, it
    checks the marginal distribution of the drawn states against the values
    computed from the smoother and prints the result.

    With the fixed simulation smoother, it prints:

    True values:
    [1.         0.66666667 0.66666667 1.        ]
    [0.         0.95238095 0.95238095 0.        ]

    Simulated values:
    [1.         0.66699187 0.66456719 1.        ]
    [0.       0.953608 0.953198 0.      ]

    Previously, it would have printed:

    True values:
    [1.         0.66666667 0.66666667 1.        ]
    [0.         0.95238095 0.95238095 0.        ]

    Simulated values:
    [1.         0.66666667 0.66666667 1.        ]
    [0. 0. 0. 0.]
    