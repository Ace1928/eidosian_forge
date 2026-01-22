import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
def _translated_compare(self, atoms1, atoms2):
    """Moves the atoms around and tries to pair up atoms, assuming any
        atoms with the same symbol are indistinguishable, and honors
        periodic boundary conditions (for example, so that an atom at
        (0.1, 0., 0.) correctly is found to be close to an atom at
        (7.9, 0., 0.) if the atoms are in an orthorhombic cell with
        x-dimension of 8. Returns dmax, the maximum distance between any
        two atoms in the optimal configuration."""
    atoms1.set_constraint()
    atoms2.set_constraint()
    for index in range(3):
        assert atoms1.pbc[index] == atoms2.pbc[index]
    least = self._get_least_common(atoms1)
    indices1 = [atom.index for atom in atoms1 if atom.symbol == least[0]]
    indices2 = [atom.index for atom in atoms2 if atom.symbol == least[0]]
    comparisons = []
    repeat = []
    for bc in atoms2.pbc:
        if bc:
            repeat.append(3)
        else:
            repeat.append(1)
    repeated = atoms2.repeat(repeat)
    moved_cell = atoms2.cell * atoms2.pbc
    for moved in moved_cell:
        repeated.translate(-moved)
    repeated.set_cell(atoms2.cell)
    for index in indices2:
        comparison = repeated.copy()
        comparison.translate(-atoms2[index].position)
        comparisons.append(comparison)
    standard = atoms1.copy()
    standard.translate(-atoms1[indices1[0]].position)
    dmaxes = []
    for comparison in comparisons:
        dmax = self._indistinguishable_compare(standard, comparison)
        dmaxes.append(dmax)
    return min(dmaxes)