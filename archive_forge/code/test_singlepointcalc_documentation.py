from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.io import read
from ase.constraints import FixAtoms
Makes sure the unconstrained forces stay that way.