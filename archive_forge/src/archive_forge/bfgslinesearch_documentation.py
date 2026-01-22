import time
import numpy as np
from numpy import eye, absolute, sqrt, isinf
from ase.utils.linesearch import LineSearch
from ase.optimize.optimize import Optimizer
Initialize hessian from old trajectory.