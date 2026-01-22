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
@staticmethod
def _reverse_convert(x: float, factor1: float, factor2: float):
    """
        Converts mixing ratio x in c1 - c2 tie line to that in
        comp1 - comp2 tie line.

        Args:
            x: Mixing ratio x in c1 - c2 tie line, a float between
                0 and 1.
            factor1: Compositional ratio between composition c1 and
                processed composition comp1. E.g., factor for
                Composition('SiO2') and Composition('O') is 2.
            factor2: Compositional ratio between composition c2 and
                processed composition comp2.

        Returns:
            Mixing ratio in comp1 - comp2 tie line, a float between 0 and 1.
        """
    return x * factor1 / ((1 - x) * factor2 + x * factor1)