from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from monty.dev import requires
from monty.json import MSONable
from scipy.interpolate import UnivariateSpline
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.units import amu_to_kg
from pymatgen.phonon.bandstructure import PhononBandStructure, PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
class GruneisenParameter(MSONable):
    """Class for Grueneisen parameters on a regular grid."""

    def __init__(self, qpoints: ArrayLike, gruneisen: ArrayLike[ArrayLike], frequencies: ArrayLike[ArrayLike], multiplicities: Sequence | None=None, structure: Structure=None, lattice: Lattice=None) -> None:
        """
        Args:
            qpoints: list of qpoints as numpy arrays, in frac_coords of the given lattice by default
            gruneisen: list of gruneisen parameters as numpy arrays, shape: (3*len(structure), len(qpoints))
            frequencies: list of phonon frequencies in THz as a numpy array with shape (3*len(structure), len(qpoints))
            multiplicities: list of multiplicities
            structure: The crystal structure (as a pymatgen Structure object) associated with the gruneisen parameters.
            lattice: The reciprocal lattice as a pymatgen Lattice object. Pymatgen uses the physics convention of
                reciprocal lattice vectors WITH a 2*pi coefficient.
        """
        self.qpoints = qpoints
        self.gruneisen = gruneisen
        self.frequencies = frequencies
        self.multiplicities = multiplicities
        self.lattice = lattice
        self.structure = structure

    def average_gruneisen(self, t: float | None=None, squared: bool=True, limit_frequencies: Literal['debye', 'acoustic'] | None=None) -> float:
        """Calculates the average of the Gruneisen based on the values on the regular grid.
        If squared is True the average will use the squared value of the Gruneisen and a squared root
        is performed on the final result.
        Values associated to negative frequencies will be ignored.
        See Scripta Materialia 129, 88 for definitions.
        Adapted from classes in abipy that have been written by Guido Petretto (UCLouvain).

        Args:
            t: the temperature at which the average Gruneisen will be evaluated. If None the acoustic Debye
                temperature is used (see acoustic_debye_temp).
            squared: if True the average is performed on the squared values of the Grueneisen.
            limit_frequencies: if None (default) no limit on the frequencies will be applied.
                Possible values are "debye" (only modes with frequencies lower than the acoustic Debye
                temperature) and "acoustic" (only the acoustic modes, i.e. the first three modes).

        Returns:
            The average Gruneisen parameter
        """
        if t is None:
            t = self.acoustic_debye_temp
        w = self.frequencies
        wdkt = w * const.tera / (const.value('Boltzmann constant in Hz/K') * t)
        exp_wdkt = np.exp(wdkt)
        cv = np.choose(w > 0, (0, const.value('Boltzmann constant in eV/K') * wdkt ** 2 * exp_wdkt / (exp_wdkt - 1) ** 2))
        gamma = self.gruneisen
        if squared:
            gamma = gamma ** 2
        if limit_frequencies == 'debye':
            acoustic_debye_freq = self.acoustic_debye_temp * const.value('Boltzmann constant in Hz/K') / const.tera
            ind = np.where((w >= 0) & (w <= acoustic_debye_freq))
        elif limit_frequencies == 'acoustic':
            w_acoustic = w[:, :3]
            ind = np.where(w_acoustic >= 0)
        elif limit_frequencies is None:
            ind = np.where(w >= 0)
        else:
            raise ValueError(f'{limit_frequencies} is not an accepted value for limit_frequencies.')
        weights = self.multiplicities
        assert weights is not None, 'Multiplicities are not defined.'
        g = np.dot(weights[ind[0]], np.multiply(cv, gamma)[ind]).sum() / np.dot(weights[ind[0]], cv[ind]).sum()
        if squared:
            g = np.sqrt(g)
        return g

    def thermal_conductivity_slack(self, squared: bool=True, limit_frequencies: Literal['debye', 'acoustic'] | None=None, theta_d: float | None=None, t: float | None=None) -> float:
        """Calculates the thermal conductivity at the acoustic Debye temperature with the Slack formula,
        using the average Gruneisen.
        Adapted from abipy.

        Args:
            squared (bool): if True the average is performed on the squared values of the Gruenisen
            limit_frequencies: if None (default) no limit on the frequencies will be applied.
                Possible values are "debye" (only modes with frequencies lower than the acoustic Debye
                temperature) and "acoustic" (only the acoustic modes, i.e. the first three modes).
            theta_d: the temperature used to estimate the average of the Gruneisen used in the
                Slack formula. If None the acoustic Debye temperature is used (see
                acoustic_debye_temp). Will also be considered as the Debye temperature in the
                Slack formula.
            t: temperature at which the thermal conductivity is estimated. If None the value at
                the calculated acoustic Debye temperature is given. The value is obtained as a
                simple rescaling of the value at the Debye temperature.

        Returns:
            The value of the thermal conductivity in W/(m*K)
        """
        assert self.structure is not None, 'Structure is not defined.'
        average_mass = np.mean([s.specie.atomic_mass for s in self.structure]) * amu_to_kg
        if theta_d is None:
            theta_d = self.acoustic_debye_temp
        mean_g = self.average_gruneisen(t=theta_d, squared=squared, limit_frequencies=limit_frequencies)
        f1 = 0.849 * 3 * 4 ** (1 / 3) / (20 * np.pi ** 3 * (1 - 0.514 * mean_g ** (-1) + 0.228 * mean_g ** (-2)))
        f2 = (const.k * theta_d / const.hbar) ** 2
        f3 = const.k * average_mass * self.structure.volume ** (1 / 3) * 1e-10 / (const.hbar * mean_g ** 2)
        k = f1 * f2 * f3
        if t is not None:
            k *= theta_d / t
        return k

    @property
    @requires(phonopy, 'This method requires phonopy to be installed')
    def tdos(self):
        """The total DOS (re)constructed from the gruneisen.yaml file."""

        class TempMesh:
            """Temporary Class."""
        a = TempMesh()
        a.frequencies = np.transpose(self.frequencies)
        a.weights = self.multiplicities
        b = TotalDos(a)
        b.run()
        return b

    @property
    def phdos(self) -> PhononDos:
        """Returns: PhononDos object."""
        return PhononDos(self.tdos.frequency_points, self.tdos.dos)

    @property
    def debye_temp_limit(self) -> float:
        """Debye temperature in K. Adapted from apipy."""
        f_mesh = self.tdos.frequency_points * const.tera
        dos = self.tdos.dos
        i_a = UnivariateSpline(f_mesh, dos * f_mesh ** 2, s=0).integral(f_mesh[0], f_mesh[-1])
        i_b = UnivariateSpline(f_mesh, dos, s=0).integral(f_mesh[0], f_mesh[-1])
        integrals = i_a / i_b
        return np.sqrt(5 / 3 * integrals) / const.value('Boltzmann constant in Hz/K')

    def debye_temp_phonopy(self, freq_max_fit=None) -> float:
        """Get Debye temperature in K as implemented in phonopy.

        Args:
            freq_max_fit: Maximum frequency to include for fitting.
                          Defaults to include first quartile of frequencies.

        Returns:
            Debye temperature in K.
        """
        assert self.structure is not None, 'Structure is not defined.'
        t = self.tdos
        t.set_Debye_frequency(num_atoms=len(self.structure), freq_max_fit=freq_max_fit)
        f_d = t.get_Debye_frequency()
        return const.value('Planck constant') * f_d * const.tera / const.value('Boltzmann constant')

    @property
    def acoustic_debye_temp(self) -> float:
        """Acoustic Debye temperature in K, i.e. the Debye temperature divided by n_sites**(1/3).
        Adapted from abipy.
        """
        assert self.structure is not None, 'Structure is not defined.'
        return self.debye_temp_limit / len(self.structure) ** (1 / 3)