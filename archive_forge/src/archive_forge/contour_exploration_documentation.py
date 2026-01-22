import numpy as np
from ase.optimize.optimize import Dynamics
Computes the potentiostat step size by linear extrapolation of the
        potential energy using the forces. The step size can be positive or
        negative depending on whether or not the energy is too high or too low.
        