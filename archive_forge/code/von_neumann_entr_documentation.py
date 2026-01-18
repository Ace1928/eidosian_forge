from typing import List, Tuple
import numpy as np
from scipy import linalg as LA
from scipy.special import entr
from cvxpy.atoms.atom import Atom
from cvxpy.constraints.constraint import Constraint
Returns constraints describing the domain of the node.
        