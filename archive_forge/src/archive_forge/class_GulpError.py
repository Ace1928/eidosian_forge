from __future__ import annotations
import os
import re
import subprocess
from monty.tempfile import ScratchDir
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class GulpError(Exception):
    """Exception class for GULP.
    Raised when the GULP gives an error.
    """

    def __init__(self, msg):
        """
        Args:
            msg (str): Message.
        """
        self.msg = msg

    def __str__(self):
        return 'GulpError : ' + self.msg