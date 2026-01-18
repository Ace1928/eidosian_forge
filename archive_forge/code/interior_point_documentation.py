from pyomo.contrib.pynumero.interfaces.utils import (
import numpy as np
import logging
import time
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus
from pyomo.common.timing import HierarchicalTimer
import enum

        Performance could be improved significantly by only performing 
        symbolic factorization once.

        However, we first have to make sure the nonzero structure 
        (and ordering of row and column arrays) of the KKT matrix never 
        changes. We have not had time to test this thoroughly, yet. 
        