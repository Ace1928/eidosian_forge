from __future__ import annotations
import logging
from collections import namedtuple
from typing import TYPE_CHECKING, Callable
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from matplotlib.collections import LineCollection
from monty.json import jsanitize
from pymatgen.electronic_structure.plotter import BSDOSPlotter, plot_brillouin_zone
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.gruneisen import GruneisenPhononBandStructureSymmLine
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig, pretty_plot
def _get_weight(self, vec: np.ndarray, indices: list[list[int]]) -> np.ndarray:
    """Compute the weight for each combination of sites according to the
        eigenvector.
        """
    num_atom = int(self.n_bands / 3)
    new_vec = np.zeros(num_atom)
    for idx in range(num_atom):
        new_vec[idx] = np.linalg.norm(vec[idx * 3:idx * 3 + 3])
    gw = []
    norm_f = 0
    for comb in indices:
        projector = np.zeros(len(new_vec))
        for idx in range(len(projector)):
            if idx in comb:
                projector[idx] = 1
        group_weight = np.dot(projector, new_vec)
        gw.append(group_weight)
        norm_f += group_weight
    return np.array(gw, dtype=float) / norm_f