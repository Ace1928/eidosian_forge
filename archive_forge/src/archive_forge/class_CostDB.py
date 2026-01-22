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
class CostDB(abc.ABC):
    """
    Abstract class for representing a Cost database.
    Can be extended, e.g. for file-based or REST-based databases.
    """

    @abc.abstractmethod
    def get_entries(self, chemsys):
        """
        For a given chemical system, return an array of CostEntries.

        Args:
            chemsys:
                array of Elements defining the chemical system.

        Returns:
            array of CostEntries
        """
        return