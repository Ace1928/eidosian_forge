from ase.build import molecule
from ase.optimize import BFGS
import pytest
from ase.calculators.calculator import CalculatorSetupError
from ase import units
from ase.atoms import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
def test_unknown_keywords(cp2k_factory):
    with pytest.raises(CalculatorSetupError):
        cp2k_factory.calc(dummy_nonexistent_keyword='hello')