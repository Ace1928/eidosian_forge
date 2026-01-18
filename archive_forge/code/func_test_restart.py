from ase.build import molecule
from ase.optimize import BFGS
import pytest
from ase.calculators.calculator import CalculatorSetupError
from ase import units
from ase.atoms import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
def test_restart(cp2k_factory, atoms):
    calc = cp2k_factory.calc()
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('test_restart')
    calc2 = cp2k_factory.calc(restart='test_restart')
    assert not calc2.calculation_required(atoms, ['energy'])