from __future__ import annotations
import os
import re
import subprocess
from monty.tempfile import ScratchDir
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class GulpConvergenceError(Exception):
    """Exception class for GULP.
    Raised when proper convergence is not reached in Mott-Littleton
    defect energy optimization procedure in GULP.
    """

    def __init__(self, msg=''):
        """
        Args:
            msg (str): Message.
        """
        self.msg = msg

    def __str__(self):
        return self.msg