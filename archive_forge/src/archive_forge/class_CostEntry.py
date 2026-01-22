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
class CostEntry(PDEntry):
    """Extends PDEntry to include a BibTeX reference and include language about cost."""

    def __init__(self, composition, cost, name, reference):
        """
        Args:
            composition:
                Composition as a pymatgen.core.structure.Composition
            cost:
                Cost (per mol, NOT per kg) of the full Composition
            name:
                Optional parameter to name the entry. Defaults to the reduced
                chemical formula as in PDEntry.
            reference:
                Reference data as BiBTeX string.
        """
        super().__init__(composition, cost, name)
        if reference and (not is_valid_bibtex(reference)):
            raise ValueError('Invalid format for cost reference! Should be BibTeX string.')
        self.reference = reference

    def __repr__(self):
        return f'CostEntry : {self.composition} with cost = {self.energy:.4f}'