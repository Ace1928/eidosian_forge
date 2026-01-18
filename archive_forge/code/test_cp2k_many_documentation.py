from ase.build import molecule
from ase.optimize import BFGS
import pytest
from ase.calculators.calculator import CalculatorSetupError
from ase import units
from ase.atoms import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
Tests for the CP2K ASE calulator.

http://www.cp2k.org
Author: Ole Schuett <ole.schuett@mat.ethz.ch>
