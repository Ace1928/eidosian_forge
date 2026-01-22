import numpy as np
from cvxpy import settings
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dgp2dcp.canonicalizers import DgpCanonMethods
Converts a DGP problem to a DCP problem.
        