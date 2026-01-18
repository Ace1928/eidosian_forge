import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

        Parameters
        ----------
        sigma: float
          The potential minimum is at  2**(1/6) * sigma, default 1.0
        epsilon: float
          The potential depth, default 1.0
        rc: float, None
          Cut-off for the NeighborList is set to 3 * sigma if None.
          The energy is upshifted to be continuous at rc.
          Default None
        ro: float, None
          Onset of cutoff function in 'smooth' mode. Defaults to 2/3 * rc.
        smooth: bool, False
          Cutoff mode. False means that the pairwise energy is simply shifted
          to be 0 at r = rc, leading to the energy going to 0 continuously,
          but the forces jumping to zero discontinuously at the cutoff.
          True means that a smooth cutoff function is multiplied to the pairwise
          energy that smoothly goes to 0 between ro and rc. Both energy and
          forces are continuous in that case.
          If smooth=True, make sure to check the tail of the forces for kinks, ro
          might have to be adjusted to avoid distorting the potential too much.

        