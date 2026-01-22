import numpy as np
import pytest
from ase.atoms import Atoms
from ase.build import bulk
from ase.calculators.calculator import all_changes
from ase.calculators.lj import LennardJones
from ase.spacegroup.symmetrize import FixSymmetry, check_symmetry, is_subgroup
from ase.optimize.precon.lbfgs import PreconLBFGS
from ase.constraints import UnitCellFilter, ExpCellFilter
class NoisyLennardJones(LennardJones):

    def __init__(self, *args, rng=None, **kwargs):
        self.rng = rng
        LennardJones.__init__(self, *args, **kwargs)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        LennardJones.calculate(self, atoms, properties, system_changes)
        if 'forces' in self.results:
            self.results['forces'] += 0.0001 * self.rng.normal(size=self.results['forces'].shape)
        if 'stress' in self.results:
            self.results['stress'] += 0.0001 * self.rng.normal(size=self.results['stress'].shape)