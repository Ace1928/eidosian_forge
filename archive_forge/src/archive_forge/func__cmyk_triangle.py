from __future__ import annotations
import copy
import itertools
import logging
import math
import typing
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Literal, cast, no_type_check
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import palettable
import scipy.interpolate as scint
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from monty.dev import requires
from monty.json import jsanitize
from pymatgen.core import Element
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.boltztrap import BoltztrapError
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.util.plotting import add_fig_kwargs, get_ax3d_fig, pretty_plot
@staticmethod
def _cmyk_triangle(ax, c_label, m_label, y_label, k_label, loc) -> None:
    """Draw an RGB triangle legend on the desired axis."""
    if loc not in range(1, 11):
        loc = 2
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    inset_ax = inset_axes(ax, width=1.5, height=1.5, loc=loc)
    mesh = 35
    x = []
    y = []
    color = []
    for c in range(mesh):
        for ye in range(mesh):
            for m in range(mesh):
                if not (c == mesh - 1 and ye == mesh - 1 and (m == mesh - 1)) and (not (c == 0 and ye == 0 and (m == 0))):
                    c1 = c / (c + ye + m)
                    ye1 = ye / (c + ye + m)
                    m1 = m / (c + ye + m)
                    x.append(0.33 * (2.0 * ye1 + c1) / (c1 + ye1 + m1))
                    y.append(0.33 * np.sqrt(3) * c1 / (c1 + ye1 + m1))
                    rc = 1 - c / (mesh - 1)
                    gc = 1 - m / (mesh - 1)
                    bc = 1 - ye / (mesh - 1)
                    color.append([rc, gc, bc])
    inset_ax.scatter(x, y, s=7, marker='.', edgecolor=color)
    inset_ax.set_xlim([-0.35, 1.0])
    inset_ax.set_ylim([-0.35, 1.0])
    common = dict(fontsize=13, family='Times New Roman')
    inset_ax.text(0.7, -0.2, m_label, **common, color=(0, 0, 0), horizontalalignment='left')
    inset_ax.text(0.325, 0.7, c_label, **common, color=(0, 0, 0), horizontalalignment='center')
    inset_ax.text(-0.05, -0.2, y_label, **common, color=(0, 0, 0), horizontalalignment='right')
    inset_ax.text(0.325, 0.22, k_label, **common, color=(1, 1, 1), horizontalalignment='center')
    inset_ax.axis('off')