from .core import LpSolver, LpSolver_CMD, subprocess, PulpSolverError
from .. import constants
import warnings
import sys
import re
def writeslxsol(self, name, *values):
    """
        Write a solution file in SLX format.
        The function can write multiple solutions to the same file, each
        solution must be passed as a list of (name,value) pairs. Solutions
        are written in the order specified and are given names "solutionN"
        where N is the index of the solution in the list.

        :param string name: file name
        :param list values: list of lists of (name,value) pairs
        """
    with open(name, 'w') as slx:
        for i, sol in enumerate(values):
            slx.write('NAME solution%d\n' % i)
            for name, value in sol:
                slx.write(f' C      {name} {value:.16f}\n')
        slx.write('ENDATA\n')