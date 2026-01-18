import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.tip3p import rOH, angleHOH, TIP3P
 energy and forces on molecule a from all other molecules.
            cutoff is based on O-O Distance. 