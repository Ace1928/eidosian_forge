import numpy as np
import scipy as sp
import cvxpy as cp
from cvxpy import trace
from cvxpy.atoms import von_neumann_entr
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.utilities.linalg import onb_for_orthogonal_complement
Enforce a lower bound of 0.9 on trace(N);
        Expect N's unspecified eigenvalue to be 0.4