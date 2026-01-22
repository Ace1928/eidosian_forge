from __future__ import annotations
import itertools
import logging
import warnings
from typing import TYPE_CHECKING
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from pymatgen.core.structure import Structure
from pymatgen.util.coord import get_angle
from pymatgen.util.string import unicodeify_spacegroup
def _get_azimuth_elev(self, miller_index):
    """
        Args:
            miller_index: viewing direction.

        Returns:
            azim, elev for plotting
        """
    if miller_index in [(0, 0, 1), (0, 0, 0, 1)]:
        return (0, 90)
    cart = self.lattice.get_cartesian_coords(miller_index)
    azim = get_angle([cart[0], cart[1], 0], (1, 0, 0))
    v = [cart[0], cart[1], 0]
    elev = get_angle(cart, v)
    return (azim, elev)