from __future__ import annotations
import json
import os
from collections import namedtuple
from fractions import Fraction
from typing import TYPE_CHECKING, cast
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.constants as sc
from pymatgen.analysis.diffraction.core import AbstractDiffractionPatternCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.string import latexify_spacegroup, unicodeify_spacegroup
def zone_axis_filter(self, points: list[tuple[int, int, int]] | np.ndarray, laue_zone: int=0) -> list[tuple[int, int, int]]:
    """
        Filters out all points that exist within the specified Laue zone according to the zone axis rule.

        Args:
            points (np.ndarray): The list of points to be filtered.
            laue_zone (int): The desired Laue zone.

        Returns:
            list of 3-tuples
        """
    if any((isinstance(n, tuple) for n in points)):
        return list(points)
    if len(points) == 0:
        return []
    filtered = np.where(np.dot(np.array(self.beam_direction), np.transpose(points)) == laue_zone)
    result = points[filtered]
    return cast(list[tuple[int, int, int]], [tuple(x) for x in result.tolist()])