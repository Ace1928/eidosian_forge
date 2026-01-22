from the a CostDB instance, for example a CSV file via CostDBCSV.
from __future__ import annotations
import abc
import csv
import itertools
import os
from collections import defaultdict
import scipy.constants as const
from monty.design_patterns import singleton
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Element
from pymatgen.util.provenance import is_valid_bibtex
@singleton
class CostDBElements(CostDBCSV):
    """Singleton object that provides the cost data for elements."""

    def __init__(self):
        """Init."""
        CostDBCSV.__init__(self, f'{module_dir}/costdb_elements.csv')