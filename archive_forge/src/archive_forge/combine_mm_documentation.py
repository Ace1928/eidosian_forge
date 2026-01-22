import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.qmmm import combine_lj_lorenz_berthelot
from ase import units
import copy
A calculator that combines two MM calculators
        (TIPnP, Counterions, ...)

        parameters:

        idx: List of indices of atoms belonging to calculator 1
        apm1,2: atoms pr molecule of each subsystem (NB: apm for TIP4P is 3!)
        calc1,2: calculator objects for each subsystem
        sig1,2, eps1,2: LJ parameters for each subsystem. Should be a numpy
                        array of length = apm
        rc = long range cutoff
        width = width of cutoff region.

        Currently the interactions are limited to being:
        - Nonbonded
        - Hardcoded to two terms:
            - Coulomb electrostatics
            - Lennard-Jones

        It could of course benefit from being more like the EIQMMM class
        where the interactions are switchable. But this is in princple
        just meant for adding counter ions to a qmmm simulation to neutralize
        the charge of the total systemn

        Maybe it can combine n MM calculators in the future?
        