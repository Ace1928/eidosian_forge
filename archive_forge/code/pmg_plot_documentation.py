from __future__ import annotations
import matplotlib.pyplot as plt
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.plotter import DosPlotter
from pymatgen.io.vasp import Chgcar, Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.plotting import pretty_plot
Master control method calling other plot methods based on args.

    Args:
        args (dict): Args from argparse.
    