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
def _get_xaxis_title(self, latex: bool=True) -> str:
    """Returns the formatted title of the x axis (using either html/latex)."""
    if latex:
        f1 = latexify(self.c1.reduced_formula)
        f2 = latexify(self.c2.reduced_formula)
        title = f'$x$ in $x${f1} + $(1-x)${f2}'
    else:
        f1 = htmlify(self.c1.reduced_formula)
        f2 = htmlify(self.c2.reduced_formula)
        title = f'<i>x</i> in <i>x</i>{f1} + (1-<i>x</i>){f2}'
    return title