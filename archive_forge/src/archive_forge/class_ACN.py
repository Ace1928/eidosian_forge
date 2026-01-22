import numpy as np
import ase.units as units
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_masses
from ase.geometry import find_mic
class ACN(Calculator):
    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, rc=5.0, width=1.0):
        """Three-site potential for acetonitrile.

        Atom sequence must be:
        MeCNMeCN ... MeCN or NCMeNCMe ... NCMe

        When performing molecular dynamics (MD), forces are redistributed
        and only Me and N sites propagated based on a scheme for MD of
        rigid triatomic molecules from Ciccotti et al. Molecular Physics
        1982 (https://doi.org/10.1080/00268978200100942). Apply constraints
        using the FixLinearTriatomic to fix the geometry of the acetonitrile
        molecules.

        rc: float
            Cutoff radius for Coulomb interactions.
        width: float
            Width for cutoff function for Coulomb interactions.

        References:

        https://doi.org/10.1080/08927020108024509
        """
        self.rc = rc
        self.width = width
        self.forces = None
        Calculator.__init__(self)
        self.sites_per_mol = 3
        self.pcpot = None

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        Z = atoms.numbers
        masses = atoms.get_masses()
        if Z[0] == 7:
            n = 0
            me = 2
            sigma = np.array([sigma_n, sigma_c, sigma_me])
            epsilon = np.array([epsilon_n, epsilon_c, epsilon_me])
        else:
            n = 2
            me = 0
            sigma = np.array([sigma_me, sigma_c, sigma_n])
            epsilon = np.array([epsilon_me, epsilon_c, epsilon_n])
        assert (Z[n::3] == 7).all(), 'incorrect atoms sequence'
        assert (Z[1::3] == 6).all(), 'incorrect atoms sequence'
        assert (masses[n::3] == m_n).all(), 'incorrect masses'
        assert (masses[1::3] == m_c).all(), 'incorrect masses'
        assert (masses[me::3] == m_me).all(), 'incorrect masses'
        R = self.atoms.positions.reshape((-1, 3, 3))
        pbc = self.atoms.pbc
        cd = self.atoms.cell.diagonal()
        nm = len(R)
        assert (self.atoms.cell == np.diag(cd)).all(), 'not orthorhombic'
        assert ((cd >= 2 * self.rc) | ~pbc).all(), 'cutoff too large'
        charges = self.get_virtual_charges(atoms[:3])
        sigma_co, epsilon_co = combine_lj_lorenz_berthelot(sigma, epsilon)
        energy = 0.0
        self.forces = np.zeros((3 * nm, 3))
        for m in range(nm - 1):
            Dmm = R[m + 1:, 1] - R[m, 1]
            Dmm_min, Dmm_min_len = find_mic(Dmm, atoms.cell, pbc)
            shift = Dmm_min - Dmm
            cut, dcut = self.cutoff(Dmm_min_len)
            for j in range(3):
                D = R[m + 1:] - R[m, j] + shift[:, np.newaxis]
                D_len2 = (D ** 2).sum(axis=2)
                D_len = D_len2 ** 0.5
                e = charges[j] * charges / D_len * k_c
                energy += np.dot(cut, e).sum()
                F = (e / D_len2 * cut[:, np.newaxis])[:, :, np.newaxis] * D
                Fmm = -(e.sum(1) * dcut / Dmm_min_len)[:, np.newaxis] * Dmm_min
                self.forces[(m + 1) * 3:] += F.reshape((-1, 3))
                self.forces[m * 3 + j] -= F.sum(axis=0).sum(axis=0)
                self.forces[(m + 1) * 3 + 1::3] += Fmm
                self.forces[m * 3 + 1] -= Fmm.sum(0)
                c6 = (sigma_co[:, j] ** 2 / D_len2) ** 3
                c12 = c6 ** 2
                e = 4 * epsilon_co[:, j] * (c12 - c6)
                energy += np.dot(cut, e).sum()
                F = (24 * epsilon_co[:, j] * (2 * c12 - c6) / D_len2 * cut[:, np.newaxis])[:, :, np.newaxis] * D
                Fmm = -(e.sum(1) * dcut / Dmm_min_len)[:, np.newaxis] * Dmm_min
                self.forces[(m + 1) * 3:] += F.reshape((-1, 3))
                self.forces[m * 3 + j] -= F.sum(axis=0).sum(axis=0)
                self.forces[(m + 1) * 3 + 1::3] += Fmm
                self.forces[m * 3 + 1] -= Fmm.sum(0)
        if self.pcpot:
            e, f = self.pcpot.calculate(np.tile(charges, nm), self.atoms.positions)
            energy += e
            self.forces += f
        self.results['energy'] = energy
        self.results['forces'] = self.forces

    def redistribute_forces(self, forces):
        return forces

    def get_molcoms(self, nm):
        molcoms = np.zeros((nm, 3))
        for m in range(nm):
            molcoms[m] = self.atoms[m * 3:(m + 1) * 3].get_center_of_mass()
        return molcoms

    def cutoff(self, d):
        x1 = d > self.rc - self.width
        x2 = d < self.rc
        x12 = np.logical_and(x1, x2)
        y = (d[x12] - self.rc + self.width) / self.width
        cut = np.zeros(len(d))
        cut[x2] = 1.0
        cut[x12] -= y ** 2 * (3.0 - 2.0 * y)
        dtdd = np.zeros(len(d))
        dtdd[x12] -= 6.0 / self.width * y * (1.0 - y)
        return (cut, dtdd)

    def embed(self, charges):
        """Embed atoms in point-charges."""
        self.pcpot = PointChargePotential(charges)
        return self.pcpot

    def check_state(self, atoms, tol=1e-15):
        system_changes = Calculator.check_state(self, atoms, tol)
        if self.pcpot and self.pcpot.mmpositions is not None:
            system_changes.append('positions')
        return system_changes

    def add_virtual_sites(self, positions):
        return positions

    def get_virtual_charges(self, atoms):
        charges = np.empty(len(atoms))
        Z = atoms.numbers
        if Z[0] == 7:
            n = 0
            me = 2
        else:
            n = 2
            me = 0
        charges[me::3] = q_me
        charges[1::3] = q_c
        charges[n::3] = q_n
        return charges