import sys
import time
import warnings
from math import cos, sin, atan, tan, degrees, pi, sqrt
from typing import Dict, Any
import numpy as np
from ase.optimize.optimize import Optimizer
from ase.parallel import world
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import IOContext
class DimerControl(MinModeControl):
    """A class that takes care of the parameters needed for a Dimer search.

    Parameters:

    eigenmode_method: str
        The name of the eigenmode search method.
    f_rot_min: float
        Size of the rotational force under which no rotation will be
        performed.
    f_rot_max: float
        Size of the rotational force under which only one rotation will be
        performed.
    max_num_rot: int
        Maximum number of rotations per optimizer step.
    trial_angle: float
        Trial angle for the finite difference estimate of the rotational
        angle in radians.
    trial_trans_step: float
        Trial step size for the MinModeTranslate optimizer.
    maximum_translation: float
        Maximum step size and forced step size when the curvature is still
        positive for the MinModeTranslate optimizer.
    cg_translation: bool
        Conjugate Gradient for the MinModeTranslate optimizer.
    use_central_forces: bool
        Only calculate the forces at one end of the dimer and extrapolate
        the forces to the other.
    dimer_separation: float
        Separation of the dimer's images.
    initial_eigenmode_method: str
        How to construct the initial eigenmode of the dimer. If an eigenmode
        is given when creating the MinModeAtoms object, this will be ignored.
        Possible choices are: 'gauss' and 'displacement'
    extrapolate_forces: bool
        When more than one rotation is performed, an extrapolation scheme can
        be used to reduce the number of force evaluations.
    displacement_method: str
        How to displace the atoms. Possible choices are 'gauss' and 'vector'.
    gauss_std: float
        The standard deviation of the gauss curve used when doing random
        displacement.
    order: int
        How many lowest eigenmodes will be inverted.
    mask: list of bool
        Which atoms will be moved during displacement.
    displacement_center: int or [float, float, float]
        The center of displacement, nearby atoms will be displaced.
    displacement_radius: float
        When choosing which atoms to displace with the *displacement_center*
        keyword, this decides how many nearby atoms to displace.
    number_of_displacement_atoms: int
        The amount of atoms near *displacement_center* to displace.

    """
    parameters = {'eigenmode_method': 'dimer', 'f_rot_min': 0.1, 'f_rot_max': 1.0, 'max_num_rot': 1, 'trial_angle': pi / 4.0, 'trial_trans_step': 0.001, 'maximum_translation': 0.1, 'cg_translation': True, 'use_central_forces': True, 'dimer_separation': 0.0001, 'initial_eigenmode_method': 'gauss', 'extrapolate_forces': False, 'displacement_method': 'gauss', 'gauss_std': 0.1, 'order': 1, 'mask': None, 'displacement_center': None, 'displacement_radius': None, 'number_of_displacement_atoms': None}

    def log(self, parameter=None):
        """Log the parameters of the eigenmode search."""
        if self.logfile is not None:
            if parameter is not None:
                l = 'DIM:CONTROL: Updated Parameter: %s = %s\n' % (parameter, str(self.get_parameter(parameter)))
            else:
                l = 'MINMODE:METHOD: Dimer\n'
                l += 'DIM:CONTROL: Search Parameters:\n'
                l += 'DIM:CONTROL: ------------------\n'
                for key in self.parameters:
                    l += 'DIM:CONTROL: %s = %s\n' % (key, str(self.get_parameter(key)))
                l += 'DIM:CONTROL: ------------------\n'
                l += 'DIM:ROT: OPT-STEP ROT-STEP CURVATURE ROT-ANGLE ' + 'ROT-FORCE\n'
            self.logfile.write(l)
            self.logfile.flush()