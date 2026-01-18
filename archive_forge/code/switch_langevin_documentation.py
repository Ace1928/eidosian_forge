import numpy as np
from ase.md.langevin import Langevin
from ase.calculators.mixing import MixedCalculator
 Return the free energy difference between calc2 and calc1, by
        integrating dH/dlam along the switching path

        Returns
        -------
        float
            Free energy difference, F2 - F1
        