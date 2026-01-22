import os
import sys
import numpy as np
from ase import units
class CrystalThermo(ThermoChem):
    """Class for calculating thermodynamic properties of a crystalline
    solid in the approximation that a lattice of N atoms behaves as a
    system of 3N independent harmonic oscillators.

    Inputs:

    phonon_DOS : list
        a list of the phonon density of states,
        where each value represents the phonon DOS at the vibrational energy
        value of the corresponding index in phonon_energies.

    phonon_energies : list
        a list of the range of vibrational energies (hbar*omega) over which
        the phonon density of states has been evaluated. This list should be
        the same length as phonon_DOS and integrating phonon_DOS over
        phonon_energies should yield approximately 3N, where N is the number
        of atoms per unit cell. If the first element of this list is
        zero-valued it will be deleted along with the first element of
        phonon_DOS. Units of vibrational energies are eV.

    potentialenergy : float
        the potential energy in eV (e.g., from atoms.get_potential_energy)
        (if potentialenergy is unspecified, then the methods of this
        class can be interpreted as the energy corrections)

    formula_units : int
        the number of formula units per unit cell. If unspecified, the
        thermodynamic quantities calculated will be listed on a
        per-unit-cell basis.
    """

    def __init__(self, phonon_DOS, phonon_energies, formula_units=None, potentialenergy=0.0):
        self.phonon_energies = phonon_energies
        self.phonon_DOS = phonon_DOS
        if formula_units:
            self.formula_units = formula_units
            self.potentialenergy = potentialenergy / formula_units
        else:
            self.formula_units = 0
            self.potentialenergy = potentialenergy

    def get_internal_energy(self, temperature, verbose=True):
        """Returns the internal energy, in eV, of crystalline solid
        at a specified temperature (K)."""
        self.verbose = verbose
        write = self._vprint
        fmt = '%-15s%13.4f eV'
        if self.formula_units == 0:
            write('Internal energy components at T = %.2f K,\non a per-unit-cell basis:' % temperature)
        else:
            write('Internal energy components at T = %.2f K,\non a per-formula-unit basis:' % temperature)
        write('=' * 31)
        U = 0.0
        omega_e = self.phonon_energies
        dos_e = self.phonon_DOS
        if omega_e[0] == 0.0:
            omega_e = np.delete(omega_e, 0)
            dos_e = np.delete(dos_e, 0)
        write(fmt % ('E_pot', self.potentialenergy))
        U += self.potentialenergy
        zpe_list = omega_e / 2.0
        if self.formula_units == 0:
            zpe = np.trapz(zpe_list * dos_e, omega_e)
        else:
            zpe = np.trapz(zpe_list * dos_e, omega_e) / self.formula_units
        write(fmt % ('E_ZPE', zpe))
        U += zpe
        B = 1.0 / (units.kB * temperature)
        E_vib = omega_e / (np.exp(omega_e * B) - 1.0)
        if self.formula_units == 0:
            E_phonon = np.trapz(E_vib * dos_e, omega_e)
        else:
            E_phonon = np.trapz(E_vib * dos_e, omega_e) / self.formula_units
        write(fmt % ('E_phonon', E_phonon))
        U += E_phonon
        write('-' * 31)
        write(fmt % ('U', U))
        write('=' * 31)
        return U

    def get_entropy(self, temperature, verbose=True):
        """Returns the entropy, in eV/K, of crystalline solid
        at a specified temperature (K)."""
        self.verbose = verbose
        write = self._vprint
        fmt = '%-15s%13.7f eV/K%13.4f eV'
        if self.formula_units == 0:
            write('Entropy components at T = %.2f K,\non a per-unit-cell basis:' % temperature)
        else:
            write('Entropy components at T = %.2f K,\non a per-formula-unit basis:' % temperature)
        write('=' * 49)
        write('%15s%13s     %13s' % ('', 'S', 'T*S'))
        omega_e = self.phonon_energies
        dos_e = self.phonon_DOS
        if omega_e[0] == 0.0:
            omega_e = np.delete(omega_e, 0)
            dos_e = np.delete(dos_e, 0)
        B = 1.0 / (units.kB * temperature)
        S_vib = omega_e / (temperature * (np.exp(omega_e * B) - 1.0)) - units.kB * np.log(1.0 - np.exp(-omega_e * B))
        if self.formula_units == 0:
            S = np.trapz(S_vib * dos_e, omega_e)
        else:
            S = np.trapz(S_vib * dos_e, omega_e) / self.formula_units
        write('-' * 49)
        write(fmt % ('S', S, S * temperature))
        write('=' * 49)
        return S

    def get_helmholtz_energy(self, temperature, verbose=True):
        """Returns the Helmholtz free energy, in eV, of crystalline solid
        at a specified temperature (K)."""
        self.verbose = True
        write = self._vprint
        U = self.get_internal_energy(temperature, verbose=verbose)
        write('')
        S = self.get_entropy(temperature, verbose=verbose)
        F = U - temperature * S
        write('')
        if self.formula_units == 0:
            write('Helmholtz free energy components at T = %.2f K,\non a per-unit-cell basis:' % temperature)
        else:
            write('Helmholtz free energy components at T = %.2f K,\non a per-formula-unit basis:' % temperature)
        write('=' * 23)
        fmt = '%5s%15.4f eV'
        write(fmt % ('U', U))
        write(fmt % ('-T*S', -temperature * S))
        write('-' * 23)
        write(fmt % ('F', F))
        write('=' * 23)
        return F