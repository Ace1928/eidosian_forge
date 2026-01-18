import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
def test_pdlp_lp_4_no_presolve(self) -> None:
    from ortools.pdlp import solvers_pb2
    params = solvers_pb2.PrimalDualHybridGradientParams()
    params.presolve_options.use_glop = False
    StandardTestLPs.test_lp_4(solver='PDLP', parameters_proto=params)