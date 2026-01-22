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
@staticmethod
def _make_color(colors: Sequence[int]) -> Sequence[int]:
    """Convert the eigen-displacements to rgb colors."""
    if len(colors) == 2:
        return [colors[0], 0, colors[1]]
    if len(colors) == 3:
        return colors
    if len(colors) == 4:
        red = (1 - colors[0]) * (1 - colors[3])
        green = (1 - colors[1]) * (1 - colors[3])
        blue = (1 - colors[2]) * (1 - colors[3])
        return [red, green, blue]
    raise ValueError(f'Expected 2, 3 or 4 colors, got {len(colors)}')