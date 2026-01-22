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
def _get_plotly_figure(self) -> Figure:
    """Returns a Plotly figure of reaction kinks diagram."""
    kinks = map(list, zip(*self.get_kinks()))
    _, x, energy, reactions, _ = kinks
    lines = Scatter(x=x, y=energy, mode='lines', name='Lines', line={'color': 'navy', 'dash': 'solid', 'width': 5.0}, hoverinfo='none')
    annotations = self._get_plotly_annotations(x, energy, reactions)
    min_idx = energy.index(min(energy))
    x_min = x.pop(min_idx)
    e_min = energy.pop(min_idx)
    rxn_min = reactions.pop(min_idx)
    labels = [f'{htmlify(str(r))} <br>ΔE<sub>rxn</sub> = {round(e, 3)} eV/atom' for r, e in zip(reactions, energy)]
    markers = Scatter(x=x, y=energy, mode='markers', name='Reactions', hoverinfo='text', hovertext=labels, marker={'color': 'black', 'size': 12, 'opacity': 0.8, 'line': {'color': 'black', 'width': 3}}, hoverlabel={'bgcolor': 'navy'})
    min_label = f'{htmlify(str(rxn_min))} <br>ΔE<sub>rxn</sub> = {round(e_min, 3)} eV/atom'
    minimum = Scatter(x=[x_min], y=[e_min], mode='markers', hoverinfo='text', hovertext=[min_label], marker={'color': 'darkred', 'size': 24, 'symbol': 'star'}, name='Suggested reaction')
    data = [lines, markers, minimum]
    layout = plotly_layouts['default_interface_rxn_layout']
    layout['xaxis']['title'] = self._get_xaxis_title(latex=False)
    layout['annotations'] = annotations
    return Figure(data=data, layout=layout)