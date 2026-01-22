from __future__ import annotations
import os
import re
import subprocess
from monty.tempfile import ScratchDir
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class TersoffPotential:
    """Generate Tersoff Potential Table from "OxideTersoffPotentialentials" file."""

    def __init__(self):
        """Init TersoffPotential."""
        with open(f'{module_dir}/OxideTersoffPotentials') as file:
            data = {}
            for row in file:
                metaloxi = row.split()[0]
                line = row.split(')')
                data[metaloxi] = line[1]
        self.data = data