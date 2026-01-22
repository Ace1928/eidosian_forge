from __future__ import annotations
import json
import os
import warnings
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
from monty.json import MSONable
from pandas import DataFrame
from plotly.graph_objects import Figure, Scatter
from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, PhaseDiagram
from pymatgen.analysis.reaction_calculator import Reaction
from pymatgen.core.composition import Composition
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import htmlify, latexify
def _get_elem_amt_in_rxn(self, rxn: Reaction) -> float:
    """
        Computes total number of atoms in a reaction formula for elements
        not in external reservoir. This method is used in the calculation
        of reaction energy per mol of reaction formula.

        Args:
            rxn: a Reaction object.

        Returns:
            Total number of atoms for non_reservoir elements.
        """
    return sum((rxn.get_el_amount(e) for e in self.pd.elements))