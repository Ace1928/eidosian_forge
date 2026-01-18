from __future__ import annotations
from tabulate import tabulate
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
Master function to handle which operation to perform.

    Args:
        args (dict): Args from argparse.
    