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
def _rescale_distances(bs_ref, bs):
    """Method to rescale distances of bs to distances in bs_ref.
        This is used for plotting two bandstructures (same k-path)
        of different materials.
        """
    scaled_distances = []
    for br, br2 in zip(bs_ref.branches, bs.branches):
        start = br['start_index']
        end = br['end_index']
        max_d = bs_ref.distance[end]
        min_d = bs_ref.distance[start]
        s2 = br2['start_index']
        e2 = br2['end_index']
        np = e2 - s2
        if np == 0:
            scaled_distances.extend([min_d])
        else:
            scaled_distances.extend([(max_d - min_d) / np * i + min_d for i in range(np + 1)])
    return scaled_distances