from __future__ import division
from typing import List, Optional, Tuple
import numpy as np
from scipy.sparse import csc_matrix
from scipy.special import rel_entr as rel_entr_scipy
from cvxpy.atoms.elementwise.elementwise import Elementwise
Returns constraints describing the domain of the node.
        