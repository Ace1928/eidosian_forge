from __future__ import annotations
import os
import tempfile
import warnings
from collections import namedtuple
from pathlib import Path
from shutil import which
from subprocess import Popen, TimeoutExpired
from monty.dev import requires
from pymatgen.core.structure import Structure
Private function to parse clusters.out file
    Args:
        path: directory to perform parsing.

    Returns:
        list[dict]: List of cluster dictionaries with keys:
            multiplicity: int
            longest_pair_length: float
            num_points_in_cluster: int
            coordinates: list[dict] of points with keys:
                coordinates: list[float]
                num_possible_species: int
                cluster_function: float
    