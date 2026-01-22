from __future__ import annotations
import re
import sys
import warnings
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from scipy.interpolate import InterpolatedUnivariateSpline
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.io.lmto import LMTOCopl
from pymatgen.io.lobster import Cohpcar
from pymatgen.util.coord import get_linear_interpolated_value
from pymatgen.util.due import Doi, due
from pymatgen.util.num import round_to_sigfigs
class IcohpValue(MSONable):
    """Class to store information on an ICOHP or ICOOP value.

    Attributes:
        energies (ndarray): Energy values for the COHP/ICOHP/COOP/ICOOP.
        densities (ndarray): Density of states values for the COHP/ICOHP/COOP/ICOOP.
        energies_are_cartesian (bool): Whether the energies are cartesian or not.
        are_coops (bool): Whether the object is a COOP/ICOOP or not.
        are_cobis (bool): Whether the object is a COBIS/ICOBIS or not.
        icohp (dict): A dictionary of the ICOHP/COHP values. The keys are Spin.up and Spin.down.
        summed_icohp (float): The summed ICOHP/COHP values.
        num_bonds (int): The number of bonds used for the average COHP (relevant for Lobster versions <3.0).
    """

    def __init__(self, label, atom1, atom2, length, translation, num, icohp, are_coops=False, are_cobis=False, orbitals=None) -> None:
        """
        Args:
            label: label for the icohp
            atom1: str of atom that is contributing to the bond
            atom2: str of second atom that is contributing to the bond
            length: float of bond lengths
            translation: translation list, e.g. [0,0,0]
            num: integer describing how often the bond exists
            icohp: dict={Spin.up: icohpvalue for spin.up, Spin.down: icohpvalue for spin.down}
            are_coops: if True, this are COOPs
            are_cobis: if True, this are COBIs
            orbitals: {[str(Orbital1)-str(Orbital2)]: {"icohp":{Spin.up: icohpvalue for spin.up, Spin.down:
                icohpvalue for spin.down}, "orbitals":[Orbital1, Orbital2]}}.
        """
        if are_coops and are_cobis:
            raise ValueError('You cannot have info about COOPs and COBIs in the same file.')
        self._are_coops = are_coops
        self._are_cobis = are_cobis
        self._label = label
        self._atom1 = atom1
        self._atom2 = atom2
        self._length = length
        self._translation = translation
        self._num = num
        self._icohp = icohp
        self._orbitals = orbitals
        if Spin.down in self._icohp:
            self._is_spin_polarized = True
        else:
            self._is_spin_polarized = False

    def __str__(self) -> str:
        """String representation of the ICOHP/ICOOP."""
        if not self._are_coops and (not self._are_cobis):
            if self._is_spin_polarized:
                return f'ICOHP {self._label} between {self._atom1} and {self._atom2} ({self._translation}): {self._icohp[Spin.up]} eV (Spin up) and {self._icohp[Spin.down]} eV (Spin down)'
            return f'ICOHP {self._label} between {self._atom1} and {self._atom2} ({self._translation}): {self._icohp[Spin.up]} eV (Spin up)'
        if self._are_coops and (not self._are_cobis):
            if self._is_spin_polarized:
                return f'ICOOP {self._label} between {self._atom1} and {self._atom2} ({self._translation}): {self._icohp[Spin.up]} eV (Spin up) and {self._icohp[Spin.down]} eV (Spin down)'
            return f'ICOOP {self._label} between {self._atom1} and {self._atom2} ({self._translation}): {self._icohp[Spin.up]} eV (Spin up)'
        if self._are_cobis and (not self._are_coops):
            if self._is_spin_polarized:
                return f'ICOBI {self._label} between {self._atom1} and {self._atom2} ({self._translation}): {self._icohp[Spin.up]} eV (Spin up) and {self._icohp[Spin.down]} eV (Spin down)'
            return f'ICOBI {self._label} between {self._atom1} and {self._atom2} ({self._translation}): {self._icohp[Spin.up]} eV (Spin up)'
        return ''

    @property
    def num_bonds(self):
        """Tells the number of bonds for which the ICOHP value is an average.

        Returns:
            Int.
        """
        return self._num

    @property
    def are_coops(self) -> bool:
        """Tells if ICOOPs or not.

        Returns:
            Boolean.
        """
        return self._are_coops

    @property
    def are_cobis(self) -> bool:
        """Tells if ICOBIs or not.

        Returns:
            Boolean.
        """
        return self._are_cobis

    @property
    def is_spin_polarized(self) -> bool:
        """Tells if spin polarized calculation or not.

        Returns:
            Boolean.
        """
        return self._is_spin_polarized

    def icohpvalue(self, spin=Spin.up):
        """
        Args:
            spin: Spin.up or Spin.down.

        Returns:
            icohpvalue (float) corresponding to chosen spin.
        """
        if not self.is_spin_polarized and spin == Spin.down:
            raise ValueError('The calculation was not performed with spin polarization')
        return self._icohp[spin]

    def icohpvalue_orbital(self, orbitals, spin=Spin.up):
        """
        Args:
            orbitals: List of Orbitals or "str(Orbital1)-str(Orbital2)"
            spin: Spin.up or Spin.down.

        Returns:
            icohpvalue (float) corresponding to chosen spin.
        """
        if not self.is_spin_polarized and spin == Spin.down:
            raise ValueError('The calculation was not performed with spin polarization')
        if isinstance(orbitals, list):
            orbitals = f'{orbitals[0]}-{orbitals[1]}'
        return self._orbitals[orbitals]['icohp'][spin]

    @property
    def icohp(self):
        """Dict with icohps for spinup and spindown
        Returns:
            dict={Spin.up: icohpvalue for spin.up, Spin.down: icohpvalue for spin.down}.
        """
        return self._icohp

    @property
    def summed_icohp(self):
        """Sums ICOHPs of both spin channels for spin polarized compounds.

        Returns:
            float: icohp value in eV.
        """
        return self._icohp[Spin.down] + self._icohp[Spin.up] if self._is_spin_polarized else self._icohp[Spin.up]

    @property
    def summed_orbital_icohp(self):
        """Sums orbitals-resolved ICOHPs of both spin channels for spin-plarized compounds.

        Returns:
            {"str(Orbital1)-str(Ortibal2)": icohp value in eV}.
        """
        orbital_icohp = {}
        for orb, item in self._orbitals.items():
            orbital_icohp[orb] = item['icohp'][Spin.up] + item['icohp'][Spin.down] if self._is_spin_polarized else item['icohp'][Spin.up]
        return orbital_icohp