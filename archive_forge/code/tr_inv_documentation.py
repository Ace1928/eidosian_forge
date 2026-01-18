from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
from numpy import linalg as LA
import cvxpy.settings as s
from cvxpy.atoms.atom import Atom
from cvxpy.constraints.constraint import Constraint
Returns constraints describing the domain of the node.
        