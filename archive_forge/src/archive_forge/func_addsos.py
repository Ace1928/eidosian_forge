from .core import LpSolver, LpSolver_CMD, subprocess, PulpSolverError
from .. import constants
import warnings
import sys
import re
def addsos(m, sosdict, sostype):
    """Extract sos constraints from PuLP."""
    soslist = []
    for name in sorted(sosdict):
        indices = []
        weights = []
        for v, val in sosdict[name].items():
            indices.append(v._xprs[0])
            weights.append(val)
        soslist.append(xpress.sos(indices, weights, sostype, str(name)))
    if len(soslist):
        m.addSOS(soslist)