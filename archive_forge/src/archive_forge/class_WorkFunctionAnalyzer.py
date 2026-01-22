from __future__ import annotations
import copy
import itertools
import random
import warnings
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol
from sympy.solvers import linsolve, solve
from pymatgen.analysis.wulff import WulffShape
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.surface import get_slab_regions
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.outputs import Locpot, Outcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
class WorkFunctionAnalyzer:
    """
    A class used for calculating the work function from a slab model and
    visualizing the behavior of the local potential along the slab.

    Attributes:
        efermi (float): The Fermi energy.
        locpot_along_c (list): Local potential in eV along points along the c axis.
        vacuum_locpot (float): The maximum local potential along the c direction for the slab model,
            i.e. the potential at the vacuum.
        work_function (float): The minimum energy needed to move an electron from the surface to infinity.
            Defined as the difference between the potential at the vacuum and the Fermi energy.
        slab (Slab): The slab structure model.
        along_c (list): Points along the c direction with same increments as the locpot in the c axis.
        ave_locpot (float): Mean of the minimum and maximum (vacuum) locpot along c.
        sorted_sites (list): List of sites from the slab sorted along the c direction.
        ave_bulk_p (float): The average locpot of the slab region along the c direction.
    """

    def __init__(self, structure: Structure, locpot_along_c, efermi, shift=0, blength=3.5):
        """
        Initializes the WorkFunctionAnalyzer class.

        Args:
            structure (Structure): Structure object modelling the surface
            locpot_along_c (list): Local potential along the c direction
            outcar (MSONable): Outcar vasp output object
            shift (float): Parameter to translate the slab (and
                therefore the vacuum) of the slab structure, thereby
                translating the plot along the x axis.
            blength (float (Ang)): The longest bond length in the material.
                Used to handle pbc for noncontiguous slab layers
        """
        if shift < 0:
            shift += -1 * int(shift) + 1
        elif shift >= 1:
            shift -= int(shift)
        self.shift = shift
        slab = structure.copy()
        slab.translate_sites([idx for idx, site in enumerate(slab)], [0, 0, self.shift])
        self.slab = slab
        self.sorted_sites = sorted(self.slab, key=lambda site: site.frac_coords[2])
        self.along_c = np.linspace(0, 1, num=len(locpot_along_c))
        locpot_along_c_mid, locpot_end, locpot_start = ([], [], [])
        for idx, s in enumerate(self.along_c):
            j = s + self.shift
            if j > 1:
                locpot_start.append(locpot_along_c[idx])
            elif j < 0:
                locpot_end.append(locpot_along_c[idx])
            else:
                locpot_along_c_mid.append(locpot_along_c[idx])
        self.locpot_along_c = locpot_start + locpot_along_c_mid + locpot_end
        self.slab_regions = get_slab_regions(self.slab, blength=blength)
        bulk_p = []
        for r in self.slab_regions:
            bulk_p.extend([pot for idx, pot in enumerate(self.locpot_along_c) if r[1] >= self.along_c[idx] > r[0]])
        if len(self.slab_regions) > 1:
            bulk_p.extend([pot for idx, pot in enumerate(self.locpot_along_c) if self.slab_regions[1][1] <= self.along_c[idx]])
            bulk_p.extend([pot for idx, pot in enumerate(self.locpot_along_c) if self.slab_regions[0][0] >= self.along_c[idx]])
        self.ave_bulk_p = np.mean(bulk_p)
        self.efermi = efermi
        self.vacuum_locpot = max(self.locpot_along_c)
        self.work_function = self.vacuum_locpot - self.efermi
        self.ave_locpot = (self.vacuum_locpot - min(self.locpot_along_c)) / 2

    def get_locpot_along_slab_plot(self, label_energies=True, plt=None, label_fontsize=10):
        """
        Returns a plot of the local potential (eV) vs the
            position along the c axis of the slab model (Ang).

        Args:
            label_energies (bool): Whether to label relevant energy
                quantities such as the work function, Fermi energy,
                vacuum locpot, bulk-like locpot
            plt (plt): Matplotlib pyplot object
            label_fontsize (float): Fontsize of labels

        Returns plt of the locpot vs c axis
        """
        plt = plt or pretty_plot(width=6, height=4)
        plt.plot(self.along_c, self.locpot_along_c, 'b--')
        xg, yg = ([], [])
        for idx, pot in enumerate(self.locpot_along_c):
            in_slab = False
            for r in self.slab_regions:
                if r[0] <= self.along_c[idx] <= r[1]:
                    in_slab = True
            if len(self.slab_regions) > 1:
                if self.along_c[idx] >= self.slab_regions[1][1]:
                    in_slab = True
                if self.along_c[idx] <= self.slab_regions[0][0]:
                    in_slab = True
            if in_slab or pot < self.ave_bulk_p:
                yg.append(self.ave_bulk_p)
                xg.append(self.along_c[idx])
            else:
                yg.append(pot)
                xg.append(self.along_c[idx])
        xg, yg = zip(*sorted(zip(xg, yg)))
        plt.plot(xg, yg, 'r', linewidth=2.5, zorder=-1)
        if label_energies:
            plt = self.get_labels(plt, label_fontsize=label_fontsize)
        plt.xlim([0, 1])
        plt.ylim([min(self.locpot_along_c), self.vacuum_locpot + self.ave_locpot * 0.2])
        plt.xlabel('Fractional coordinates ($\\hat{c}$)', fontsize=25)
        plt.xticks(fontsize=15, rotation=45)
        plt.ylabel('Potential (eV)', fontsize=25)
        plt.yticks(fontsize=15)
        return plt

    def get_labels(self, plt, label_fontsize=10):
        """
        Handles the optional labelling of the plot with relevant quantities

        Args:
            plt (plt): Plot of the locpot vs c axis
            label_fontsize (float): Fontsize of labels
        Returns Labelled plt.
        """
        if len(self.slab_regions) > 1:
            label_in_vac = (self.slab_regions[0][1] + self.slab_regions[1][0]) / 2
            if abs(self.slab_regions[0][0] - self.slab_regions[0][1]) > abs(self.slab_regions[1][0] - self.slab_regions[1][1]):
                label_in_bulk = self.slab_regions[0][1] / 2
            else:
                label_in_bulk = (self.slab_regions[1][1] + self.slab_regions[1][0]) / 2
        else:
            label_in_bulk = (self.slab_regions[0][0] + self.slab_regions[0][1]) / 2
            if self.slab_regions[0][0] > 1 - self.slab_regions[0][1]:
                label_in_vac = self.slab_regions[0][0] / 2
            else:
                label_in_vac = (1 + self.slab_regions[0][1]) / 2
        plt.plot([0, 1], [self.vacuum_locpot] * 2, 'b--', zorder=-5, linewidth=1)
        xy = [label_in_bulk, self.vacuum_locpot + self.ave_locpot * 0.05]
        plt.annotate(f'$V_{{vac}}={self.vacuum_locpot:.2f}$', xy=xy, xytext=xy, color='b', fontsize=label_fontsize)
        plt.plot([0, 1], [self.efermi] * 2, 'g--', zorder=-5, linewidth=3)
        xy = [label_in_bulk, self.efermi + self.ave_locpot * 0.05]
        plt.annotate(f'$E_F={self.efermi:.2f}$', xytext=xy, xy=xy, fontsize=label_fontsize, color='g')
        plt.plot([0, 1], [self.ave_bulk_p] * 2, 'r--', linewidth=1.0, zorder=-1)
        xy = [label_in_vac, self.ave_bulk_p + self.ave_locpot * 0.05]
        plt.annotate(f'$V^{{interior}}_{{slab}}={self.ave_bulk_p:.2f}$', xy=xy, xytext=xy, color='r', fontsize=label_fontsize)
        plt.plot([label_in_vac] * 2, [self.efermi, self.vacuum_locpot], 'k--', zorder=-5, linewidth=2)
        xy = [label_in_vac, self.efermi + self.ave_locpot * 0.05]
        plt.annotate(f'$\\Phi={self.work_function:.2f}$', xy=xy, xytext=xy, fontsize=label_fontsize)
        return plt

    def is_converged(self, min_points_frac=0.015, tol: float=0.0025):
        """
        A well converged work function should have a flat electrostatic
            potential within some distance (min_point) about where the peak
            electrostatic potential is found along the c direction of the
            slab. This is dependent on the size of the slab.

        Args:
            min_point (fractional coordinates): The number of data points
                +/- the point of where the electrostatic potential is at
                its peak along the c direction.
            tol (float): If the electrostatic potential stays the same
                within this tolerance, within the min_points, it is converged.

        Returns a bool (whether or not the work function is converged)
        """
        conv_within = tol * (max(self.locpot_along_c) - min(self.locpot_along_c))
        min_points = int(min_points_frac * len(self.locpot_along_c))
        peak_i = self.locpot_along_c.index(self.vacuum_locpot)
        all_flat = []
        for i in range(len(self.along_c)):
            if peak_i - min_points < i < peak_i + min_points:
                if abs(self.vacuum_locpot - self.locpot_along_c[i]) > conv_within:
                    all_flat.append(False)
                else:
                    all_flat.append(True)
        return all(all_flat)

    @classmethod
    def from_files(cls, poscar_filename, locpot_filename, outcar_filename, shift=0, blength=3.5) -> Self:
        """
        Initializes a WorkFunctionAnalyzer from POSCAR, LOCPOT, and OUTCAR files.

        Args:
            poscar_filename (str): The path to the POSCAR file.
            locpot_filename (str): The path to the LOCPOT file.
            outcar_filename (str): The path to the OUTCAR file.
            shift (float): The shift value. Defaults to 0.
            blength (float): The longest bond length in the material.
                Used to handle pbc for noncontiguous slab layers. Defaults to 3.5.

        Returns:
            WorkFunctionAnalyzer: A WorkFunctionAnalyzer instance.
        """
        locpot = Locpot.from_file(locpot_filename)
        outcar = Outcar(outcar_filename)
        return cls(Structure.from_file(poscar_filename), locpot.get_average_along_axis(2), outcar.efermi, shift=shift, blength=blength)