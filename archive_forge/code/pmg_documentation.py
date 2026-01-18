from __future__ import annotations
import argparse
import itertools
from tabulate import tabulate, tabulate_formats
from pymatgen.cli.pmg_analyze import analyze
from pymatgen.cli.pmg_config import configure_pmg
from pymatgen.cli.pmg_plot import plot
from pymatgen.cli.pmg_potcar import generate_potcar
from pymatgen.cli.pmg_structure import analyze_structures
from pymatgen.core import SETTINGS
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Incar, Potcar
Handle main.