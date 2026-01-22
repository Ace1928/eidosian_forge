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
class WulffFacet:
    """Helper container for each Wulff plane."""

    def __init__(self, normal, e_surf, normal_pt, dual_pt, index, m_ind_orig, miller):
        """
        Args:
            normal:
            e_surf:
            normal_pt:
            dual_pt:
            index:
            m_ind_orig:
            miller:
        """
        self.normal = normal
        self.e_surf = e_surf
        self.normal_pt = normal_pt
        self.dual_pt = dual_pt
        self.index = index
        self.m_ind_orig = m_ind_orig
        self.miller = miller
        self.points = []
        self.outer_lines = []