from __future__ import annotations
import functools
import warnings
from collections import namedtuple
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.json import MSONable
from scipy.constants import value as _cd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from pymatgen.core import Structure, get_el_sp
from pymatgen.core.spectrum import Spectrum
from pymatgen.electronic_structure.core import Orbital, OrbitalType, Spin
from pymatgen.util.coord import get_linear_interpolated_value
class Dos(MSONable):
    """Basic DOS object. All other DOS objects are extended versions of this
    object.

    Attributes:
        energies (Sequence[float]): The sequence of energies.
        densities (dict[Spin, Sequence[float]]): A dict of spin densities, e.g., {Spin.up: [...], Spin.down: [...]}.
        efermi (float): Fermi level.
    """

    def __init__(self, efermi: float, energies: ArrayLike, densities: Mapping[Spin, ArrayLike], norm_vol: float | None=None) -> None:
        """
        Args:
            efermi: Fermi level energy
            energies: A sequences of energies
            densities (dict[Spin: np.array]): representing the density of states for each Spin.
            norm_vol: The volume used to normalize the densities. Defaults to 1 if None which will not perform any
                normalization. If not None, the resulting density will have units of states/eV/Angstrom^3, otherwise
                the density will be in states/eV.
        """
        self.efermi = efermi
        self.energies = np.array(energies)
        self.norm_vol = norm_vol
        vol = norm_vol or 1
        self.densities = {k: np.array(d) / vol for k, d in densities.items()}

    def get_densities(self, spin: Spin | None=None):
        """Returns the density of states for a particular spin.

        Args:
            spin: Spin

        Returns:
            Returns the density of states for a particular spin. If Spin is
            None, the sum of all spins is returned.
        """
        if self.densities is None:
            result = None
        elif spin is None:
            if Spin.down in self.densities:
                result = self.densities[Spin.up] + self.densities[Spin.down]
            else:
                result = self.densities[Spin.up]
        else:
            result = self.densities[spin]
        return result

    def get_smeared_densities(self, sigma: float):
        """Returns the Dict representation of the densities, {Spin: densities},
        but with a Gaussian smearing of std dev sigma.

        Args:
            sigma: Std dev of Gaussian smearing function.

        Returns:
            Dict of Gaussian-smeared densities.
        """
        smeared_dens = {}
        diff = [self.energies[i + 1] - self.energies[i] for i in range(len(self.energies) - 1)]
        avg_diff = sum(diff) / len(diff)
        for spin, dens in self.densities.items():
            smeared_dens[spin] = gaussian_filter1d(dens, sigma / avg_diff)
        return smeared_dens

    def __add__(self, other):
        """Adds two DOS together. Checks that energy scales are the same.
        Otherwise, a ValueError is thrown.

        Args:
            other: Another DOS object.

        Returns:
            Sum of the two DOSs.
        """
        if not all(np.equal(self.energies, other.energies)):
            raise ValueError('Energies of both DOS are not compatible!')
        densities = {spin: self.densities[spin] + other.densities[spin] for spin in self.densities}
        return Dos(self.efermi, self.energies, densities)

    def get_interpolated_value(self, energy: float) -> dict[Spin, float]:
        """Returns interpolated density for a particular energy.

        Args:
            energy (float): Energy to return the density for.

        Returns:
            dict[Spin, float]: Density for energy for each spin.
        """
        energies = {}
        for spin in self.densities:
            energies[spin] = get_linear_interpolated_value(self.energies, self.densities[spin], energy)
        return energies

    def get_interpolated_gap(self, tol: float=0.001, abs_tol: bool=False, spin: Spin | None=None) -> tuple[float, float, float]:
        """Expects a DOS object and finds the gap.

        Args:
            tol: tolerance in occupations for determining the gap
            abs_tol: Set to True for an absolute tolerance and False for a
                relative one.
            spin: Possible values are None - finds the gap in the summed
                densities, Up - finds the gap in the up spin channel,
                Down - finds the gap in the down spin channel.

        Returns:
            tuple[float, float, float]: Energies in eV corresponding to the band gap, cbm and vbm.
        """
        tdos = self.get_densities(spin)
        if not abs_tol:
            tol = tol * tdos.sum() / tdos.shape[0]
        energies = self.energies
        below_fermi = [i for i in range(len(energies)) if energies[i] < self.efermi and tdos[i] > tol]
        above_fermi = [i for i in range(len(energies)) if energies[i] > self.efermi and tdos[i] > tol]
        vbm_start = max(below_fermi)
        cbm_start = min(above_fermi)
        if vbm_start == cbm_start:
            return (0.0, self.efermi, self.efermi)
        terminal_dens = tdos[vbm_start:vbm_start + 2][::-1]
        terminal_energies = energies[vbm_start:vbm_start + 2][::-1]
        start = get_linear_interpolated_value(terminal_dens, terminal_energies, tol)
        terminal_dens = tdos[cbm_start - 1:cbm_start + 1]
        terminal_energies = energies[cbm_start - 1:cbm_start + 1]
        end = get_linear_interpolated_value(terminal_dens, terminal_energies, tol)
        return (end - start, end, start)

    def get_cbm_vbm(self, tol: float=0.001, abs_tol: bool=False, spin: Spin | None=None) -> tuple[float, float]:
        """Expects a DOS object and finds the cbm and vbm.

        Args:
            tol: tolerance in occupations for determining the gap
            abs_tol: An absolute tolerance (True) and a relative one (False)
            spin: Possible values are None - finds the gap in the summed
                densities, Up - finds the gap in the up spin channel,
                Down - finds the gap in the down spin channel.

        Returns:
            tuple[float, float]: Energies in eV corresponding to the cbm and vbm.
        """
        tdos = self.get_densities(spin)
        if not abs_tol:
            tol = tol * tdos.sum() / tdos.shape[0]
        i_fermi = 0
        while self.energies[i_fermi] <= self.efermi:
            i_fermi += 1
        i_gap_start = i_fermi
        while i_gap_start - 1 >= 0 and tdos[i_gap_start - 1] <= tol:
            i_gap_start -= 1
        i_gap_end = i_gap_start
        while i_gap_end < tdos.shape[0] and tdos[i_gap_end] <= tol:
            i_gap_end += 1
        i_gap_end -= 1
        return (self.energies[i_gap_end], self.energies[i_gap_start])

    def get_gap(self, tol: float=0.001, abs_tol: bool=False, spin: Spin | None=None):
        """Expects a DOS object and finds the gap.

        Args:
            tol: tolerance in occupations for determining the gap
            abs_tol: An absolute tolerance (True) and a relative one (False)
            spin: Possible values are None - finds the gap in the summed
                densities, Up - finds the gap in the up spin channel,
                Down - finds the gap in the down spin channel.

        Returns:
            gap in eV
        """
        cbm, vbm = self.get_cbm_vbm(tol, abs_tol, spin)
        return max(cbm - vbm, 0.0)

    def __str__(self) -> str:
        """Returns a string which can be easily plotted (using gnuplot)."""
        if Spin.down in self.densities:
            str_arr = [f'#{'Energy':30s} {'DensityUp':30s} {'DensityDown':30s}']
            for i, energy in enumerate(self.energies):
                str_arr.append(f'{energy:.5f} {self.densities[Spin.up][i]:.5f} {self.densities[Spin.down][i]:.5f}')
        else:
            str_arr = [f'#{'Energy':30s} {'DensityUp':30s}']
            for i, energy in enumerate(self.energies):
                str_arr.append(f'{energy:.5f} {self.densities[Spin.up][i]:.5f}')
        return '\n'.join(str_arr)

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Returns Dos object from dict representation of Dos."""
        return cls(dct['efermi'], dct['energies'], {Spin(int(k)): v for k, v in dct['densities'].items()})

    def as_dict(self) -> dict:
        """JSON-serializable dict representation of Dos."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'efermi': self.efermi, 'energies': self.energies.tolist(), 'densities': {str(spin): dens.tolist() for spin, dens in self.densities.items()}}