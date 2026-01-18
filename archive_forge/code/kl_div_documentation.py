from __future__ import division
from typing import List, Optional, Tuple
import numpy as np
from scipy.sparse import csc_matrix
from scipy.special import kl_div as kl_div_scipy
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.constraints.constraint import Constraint
Returns constraints describing the domain of the node.
        