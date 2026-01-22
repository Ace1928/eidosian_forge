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
def _parse_clusters(filename):
    """Private function to parse clusters.out file
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
    """
    with open(filename) as file:
        lines = file.readlines()
    clusters = []
    cluster_block = []
    for line in lines:
        line = line.split('\n')[0]
        if line == '':
            clusters.append(cluster_block)
            cluster_block = []
        else:
            cluster_block.append(line)
    cluster_dicts = []
    for cluster in clusters:
        cluster_dict = {'multiplicity': int(cluster[0]), 'longest_pair_length': float(cluster[1]), 'num_points_in_cluster': int(cluster[2])}
        points = []
        for point in range(cluster_dict['num_points_in_cluster']):
            line = cluster[3 + point].split(' ')
            point_dict = {}
            point_dict['coordinates'] = [float(line) for line in line[0:3]]
            point_dict['num_possible_species'] = int(line[3]) + 2
            point_dict['cluster_function'] = float(line[4])
            points.append(point_dict)
        cluster_dict['coordinates'] = points
        cluster_dicts.append(cluster_dict)
    return cluster_dicts