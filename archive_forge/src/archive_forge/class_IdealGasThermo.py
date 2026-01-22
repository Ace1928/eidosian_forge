import os
import sys
import numpy as np
from ase import units
class IdealGasThermo(ThermoChem):
    """Class for calculating thermodynamic properties of a molecule
    based on statistical mechanical treatments in the ideal gas
    approximation.

    Inputs for enthalpy calculations:

    vib_energies : list
        a list of the vibrational energies of the molecule (e.g., from
        ase.vibrations.Vibrations.get_energies). The number of vibrations
        used is automatically calculated by the geometry and the number of
        atoms. If more are specified than are needed, then the lowest
        numbered vibrations are neglected. If either atoms or natoms is
        unspecified, then uses the entire list. Units are eV.
    geometry : 'monatomic', 'linear', or 'nonlinear'
        geometry of the molecule
    potentialenergy : float
        the potential energy in eV (e.g., from atoms.get_potential_energy)
        (if potentialenergy is unspecified, then the methods of this
        class can be interpreted as the energy corrections)
    natoms : integer
        the number of atoms, used along with 'geometry' to determine how
        many vibrations to use. (Not needed if an atoms object is supplied
        in 'atoms' or if the user desires the entire list of vibrations
        to be used.)

    Extra inputs needed for entropy / free energy calculations:

    atoms : an ASE atoms object
        used to calculate rotational moments of inertia and molecular mass
    symmetrynumber : integer
        symmetry number of the molecule. See, for example, Table 10.1 and
        Appendix B of C. Cramer "Essentials of Computational Chemistry",
        2nd Ed.
    spin : float
        the total electronic spin. (0 for molecules in which all electrons
        are paired, 0.5 for a free radical with a single unpaired electron,
        1.0 for a triplet with two unpaired electrons, such as O_2.)
    """

    def __init__(self, vib_energies, geometry, potentialenergy=0.0, atoms=None, symmetrynumber=None, spin=None, natoms=None):
        self.potentialenergy = potentialenergy
        self.geometry = geometry
        self.atoms = atoms
        self.sigma = symmetrynumber
        self.spin = spin
        if natoms is None:
            if atoms:
                natoms = len(atoms)
        if natoms:
            if geometry == 'nonlinear':
                self.vib_energies = vib_energies[-(3 * natoms - 6):]
            elif geometry == 'linear':
                self.vib_energies = vib_energies[-(3 * natoms - 5):]
            elif geometry == 'monatomic':
                self.vib_energies = []
        else:
            self.vib_energies = vib_energies
        if sum(np.iscomplex(self.vib_energies)):
            raise ValueError('Imaginary frequencies are present.')
        else:
            self.vib_energies = np.real(self.vib_energies)
        self.referencepressure = 100000.0

    def get_enthalpy(self, temperature, verbose=True):
        """Returns the enthalpy, in eV, in the ideal gas approximation
        at a specified temperature (K)."""
        self.verbose = verbose
        write = self._vprint
        fmt = '%-15s%13.3f eV'
        write('Enthalpy components at T = %.2f K:' % temperature)
        write('=' * 31)
        H = 0.0
        write(fmt % ('E_pot', self.potentialenergy))
        H += self.potentialenergy
        zpe = self.get_ZPE_correction()
        write(fmt % ('E_ZPE', zpe))
        H += zpe
        Cv_t = 3.0 / 2.0 * units.kB
        write(fmt % ('Cv_trans (0->T)', Cv_t * temperature))
        H += Cv_t * temperature
        if self.geometry == 'nonlinear':
            Cv_r = 3.0 / 2.0 * units.kB
        elif self.geometry == 'linear':
            Cv_r = units.kB
        elif self.geometry == 'monatomic':
            Cv_r = 0.0
        write(fmt % ('Cv_rot (0->T)', Cv_r * temperature))
        H += Cv_r * temperature
        dH_v = self._vibrational_energy_contribution(temperature)
        write(fmt % ('Cv_vib (0->T)', dH_v))
        H += dH_v
        Cp_corr = units.kB * temperature
        write(fmt % ('(C_v -> C_p)', Cp_corr))
        H += Cp_corr
        write('-' * 31)
        write(fmt % ('H', H))
        write('=' * 31)
        return H

    def get_entropy(self, temperature, pressure, verbose=True):
        """Returns the entropy, in eV/K, in the ideal gas approximation
        at a specified temperature (K) and pressure (Pa)."""
        if self.atoms is None or self.sigma is None or self.spin is None:
            raise RuntimeError('atoms, symmetrynumber, and spin must be specified for entropy and free energy calculations.')
        self.verbose = verbose
        write = self._vprint
        fmt = '%-15s%13.7f eV/K%13.3f eV'
        write('Entropy components at T = %.2f K and P = %.1f Pa:' % (temperature, pressure))
        write('=' * 49)
        write('%15s%13s     %13s' % ('', 'S', 'T*S'))
        S = 0.0
        mass = sum(self.atoms.get_masses()) * units._amu
        S_t = (2 * np.pi * mass * units._k * temperature / units._hplanck ** 2) ** (3.0 / 2)
        S_t *= units._k * temperature / self.referencepressure
        S_t = units.kB * (np.log(S_t) + 5.0 / 2.0)
        write(fmt % ('S_trans (1 bar)', S_t, S_t * temperature))
        S += S_t
        if self.geometry == 'monatomic':
            S_r = 0.0
        elif self.geometry == 'nonlinear':
            inertias = self.atoms.get_moments_of_inertia() * units._amu / (10.0 ** 10) ** 2
            S_r = np.sqrt(np.pi * np.product(inertias)) / self.sigma
            S_r *= (8.0 * np.pi ** 2 * units._k * temperature / units._hplanck ** 2) ** (3.0 / 2.0)
            S_r = units.kB * (np.log(S_r) + 3.0 / 2.0)
        elif self.geometry == 'linear':
            inertias = self.atoms.get_moments_of_inertia() * units._amu / (10.0 ** 10) ** 2
            inertia = max(inertias)
            S_r = 8 * np.pi ** 2 * inertia * units._k * temperature / self.sigma / units._hplanck ** 2
            S_r = units.kB * (np.log(S_r) + 1.0)
        write(fmt % ('S_rot', S_r, S_r * temperature))
        S += S_r
        S_e = units.kB * np.log(2 * self.spin + 1)
        write(fmt % ('S_elec', S_e, S_e * temperature))
        S += S_e
        S_v = self._vibrational_entropy_contribution(temperature)
        write(fmt % ('S_vib', S_v, S_v * temperature))
        S += S_v
        S_p = -units.kB * np.log(pressure / self.referencepressure)
        write(fmt % ('S (1 bar -> P)', S_p, S_p * temperature))
        S += S_p
        write('-' * 49)
        write(fmt % ('S', S, S * temperature))
        write('=' * 49)
        return S

    def get_gibbs_energy(self, temperature, pressure, verbose=True):
        """Returns the Gibbs free energy, in eV, in the ideal gas
        approximation at a specified temperature (K) and pressure (Pa)."""
        self.verbose = verbose
        write = self._vprint
        H = self.get_enthalpy(temperature, verbose=verbose)
        write('')
        S = self.get_entropy(temperature, pressure, verbose=verbose)
        G = H - temperature * S
        write('')
        write('Free energy components at T = %.2f K and P = %.1f Pa:' % (temperature, pressure))
        write('=' * 23)
        fmt = '%5s%15.3f eV'
        write(fmt % ('H', H))
        write(fmt % ('-T*S', -temperature * S))
        write('-' * 23)
        write(fmt % ('G', G))
        write('=' * 23)
        return G