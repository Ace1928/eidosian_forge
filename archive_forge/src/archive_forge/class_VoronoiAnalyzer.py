from __future__ import annotations
import collections
import itertools
from math import acos, pi
from typing import TYPE_CHECKING
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from pymatgen.analysis.local_env import JmolNN, VoronoiNN
from pymatgen.core import Composition, Element, PeriodicSite, Species
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class VoronoiAnalyzer:
    """
    Performs a statistical analysis of Voronoi polyhedra around each site.
    Each Voronoi polyhedron is described using Schaefli notation.
    That is a set of indices {c_i} where c_i is the number of faces with i
    number of vertices. E.g. for a bcc crystal, there is only one polyhedron
    notation of which is [0,6,0,8,0,0,...].
    In perfect crystals, these also corresponds to the Wigner-Seitz cells.
    For distorted-crystals, liquids or amorphous structures, rather than one-type,
    there is a statistical distribution of polyhedra.
    See ref: Microstructure and its relaxation in Fe-B amorphous system
    simulated by molecular dynamics,
        Stepanyuk et al., J. Non-cryst. Solids (1993), 159, 80-87.
    """

    def __init__(self, cutoff=5.0, qhull_options='Qbb Qc Qz'):
        """
        Args:
            cutoff (float): cutoff distance to search for neighbors of a given atom
                (default = 5.0)
            qhull_options (str): options to pass to qhull (optional).
        """
        self.cutoff = cutoff
        self.qhull_options = qhull_options

    def analyze(self, structure: Structure, n=0):
        """
        Performs Voronoi analysis and returns the polyhedra around atom n
        in Schlaefli notation.

        Args:
            structure (Structure): structure to analyze
            n (int): index of the center atom in structure

        Returns:
            voronoi index of n: <c3,c4,c6,c6,c7,c8,c9,c10>
                where c_i denotes number of facets with i vertices.
        """
        center = structure[n]
        neighbors = structure.get_sites_in_sphere(center.coords, self.cutoff)
        neighbors = [i[0] for i in sorted(neighbors, key=lambda s: s[1])]
        qvoronoi_input = np.array([s.coords for s in neighbors])
        voro = Voronoi(qvoronoi_input, qhull_options=self.qhull_options)
        vor_index = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        for key in voro.ridge_dict:
            if 0 in key:
                if -1 in key:
                    raise ValueError('Cutoff too short.')
                try:
                    vor_index[len(voro.ridge_dict[key]) - 3] += 1
                except IndexError:
                    pass
        return vor_index

    def analyze_structures(self, structures, step_freq=10, most_frequent_polyhedra=15):
        """
        Perform Voronoi analysis on a list of Structures.
        Note that this might take a significant amount of time depending on the
        size and number of structures.

        Args:
            structures (list): list of Structures
            cutoff (float: cutoff distance around an atom to search for
                neighbors
            step_freq (int): perform analysis every step_freq steps
            qhull_options (str): options to pass to qhull
            most_frequent_polyhedra (int): this many unique polyhedra with
                highest frequencies is stored.

        Returns:
            A list of tuples in the form (voronoi_index,frequency)
        """
        voro_dict = {}
        step = 0
        for structure in structures:
            step += 1
            if step % step_freq != 0:
                continue
            v = []
            for n in range(len(structure)):
                v.append(str(self.analyze(structure, n=n).view()))
            for voro in v:
                if voro in voro_dict:
                    voro_dict[voro] += 1
                else:
                    voro_dict[voro] = 1
        return sorted(voro_dict.items(), key=lambda x: (x[1], x[0]), reverse=True)[:most_frequent_polyhedra]

    @staticmethod
    def plot_vor_analysis(voronoi_ensemble: list[tuple[str, float]]) -> plt.Axes:
        """Plot the Voronoi analysis.

        Args:
            voronoi_ensemble (list[tuple[str, float]]): List of tuples containing labels and
                values for Voronoi analysis.

        Returns:
            plt.Axes: Matplotlib Axes object with the plotted Voronoi analysis.
        """
        labels, val = zip(*voronoi_ensemble)
        arr = np.array(val, dtype=float)
        arr /= np.sum(arr)
        pos = np.arange(len(arr)) + 0.5
        _fig, ax = plt.subplots()
        ax.barh(pos, arr, align='center', alpha=0.5)
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        ax.set(title='Voronoi Spectra', xlabel='Count')
        ax.grid(visible=True)
        return ax