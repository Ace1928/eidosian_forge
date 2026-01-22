import inspect
import json
import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import atoms_too_close, gather_atoms_by_tag
from scipy.spatial.distance import cdist
class PairwiseHarmonicPotential:
    """Parent class for interatomic potentials of the type
    E(r_ij) = 0.5 * k_ij * (r_ij - r0_ij) ** 2
    """

    def __init__(self, atoms, rcut=10.0):
        self.atoms = atoms
        self.pos0 = atoms.get_positions()
        self.rcut = rcut
        nat = len(self.atoms)
        self.nl = NeighborList([self.rcut / 2.0] * nat, skin=0.0, bothways=True, self_interaction=False)
        self.nl.update(self.atoms)
        self.calculate_force_constants()

    def calculate_force_constants(self):
        msg = "Child class needs to define a calculate_force_constants() method which computes the force constants and stores them in self.force_constants (as a list which contains, for every atom, a list of the atom's force constants with its neighbors."
        raise NotImplementedError(msg)

    def get_forces(self, atoms):
        pos = atoms.get_positions()
        cell = atoms.get_cell()
        forces = np.zeros_like(pos)
        for i, p in enumerate(pos):
            indices, offsets = self.nl.get_neighbors(i)
            p = pos[indices] + np.dot(offsets, cell)
            r = cdist(p, [pos[i]])
            v = (p - pos[i]) / r
            p0 = self.pos0[indices] + np.dot(offsets, cell)
            r0 = cdist(p0, [self.pos0[i]])
            dr = r - r0
            forces[i] = np.dot(self.force_constants[i].T, dr * v)
        return forces