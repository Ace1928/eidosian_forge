import argparse
import os
import numpy as np
import pytest
from _pytest.runner import pytest_runtest_makereport as orig_pytest_runtest_makereport
import pennylane as qml
def _tol(shots):
    if shots is None:
        return float(os.environ.get('TOL', TOL))
    return TOL_STOCHASTIC