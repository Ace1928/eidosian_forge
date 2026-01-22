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
class DimerEigenmodeSearch:
    """An implementation of the Dimer's minimum eigenvalue mode search.

    This class implements the rotational part of the dimer saddle point
    searching method.

    Parameters:

    atoms: MinModeAtoms object
        MinModeAtoms is an extension to the Atoms object, which includes
        information about the lowest eigenvalue mode.
    control: DimerControl object
        Contains the parameters necessary for the eigenmode search.
        If no control object is supplied a default DimerControl
        will be created and used.
    basis: list of xyz-values
        Eigenmode. Must be an ndarray of shape (n, 3).
        It is possible to constrain the eigenmodes to be orthogonal
        to this given eigenmode.

    Notes:

    The code is inspired, with permission, by code written by the Henkelman
    group, which can be found at http://theory.cm.utexas.edu/vtsttools/code/

    References:

    * Henkelman and Jonsson, JCP 111, 7010 (1999)
    * Olsen, Kroes, Henkelman, Arnaldsson, and Jonsson, JCP 121,
      9776 (2004).
    * Heyden, Bell, and Keil, JCP 123, 224101 (2005).
    * Kastner and Sherwood, JCP 128, 014106 (2008).

    """

    def __init__(self, atoms, control=None, eigenmode=None, basis=None, **kwargs):
        if hasattr(atoms, 'get_eigenmode'):
            self.atoms = atoms
        else:
            e = 'The atoms object must be a MinModeAtoms object'
            raise TypeError(e)
        self.basis = basis
        if eigenmode is None:
            self.eigenmode = self.atoms.get_eigenmode()
        else:
            self.eigenmode = eigenmode
        if control is None:
            self.control = DimerControl(**kwargs)
            w = 'Missing control object in ' + self.__class__.__name__ + '. Using default: DimerControl()'
            warnings.warn(w, UserWarning)
            if self.control.logfile is not None:
                self.control.logfile.write('DIM:WARN: ' + w + '\n')
                self.control.logfile.flush()
        else:
            self.control = control
            for key in kwargs:
                e = "__init__() got an unexpected keyword argument '%s'" % key
                raise TypeError(e)
        self.dR = self.control.get_parameter('dimer_separation')
        self.logfile = self.control.get_logfile()

    def converge_to_eigenmode(self):
        """Perform an eigenmode search."""
        self.set_up_for_eigenmode_search()
        stoprot = False
        f_rot_min = self.control.get_parameter('f_rot_min')
        f_rot_max = self.control.get_parameter('f_rot_max')
        trial_angle = self.control.get_parameter('trial_angle')
        max_num_rot = self.control.get_parameter('max_num_rot')
        extrapolate = self.control.get_parameter('extrapolate_forces')
        while not stoprot:
            if self.forces1E is None:
                self.update_virtual_forces()
            else:
                self.update_virtual_forces(extrapolated_forces=True)
            self.forces1A = self.forces1
            self.update_curvature()
            f_rot_A = self.get_rotational_force()
            if norm(f_rot_A) <= f_rot_min:
                self.log(f_rot_A, None)
                stoprot = True
            else:
                n_A = self.eigenmode
                rot_unit_A = normalize(f_rot_A)
                c0 = self.get_curvature()
                c0d = np.vdot(self.forces2 - self.forces1, rot_unit_A) / self.dR
                n_B, rot_unit_B = rotate_vectors(n_A, rot_unit_A, trial_angle)
                self.eigenmode = n_B
                self.update_virtual_forces()
                self.forces1B = self.forces1
                c1d = np.vdot(self.forces2 - self.forces1, rot_unit_B) / self.dR
                a1 = c0d * cos(2 * trial_angle) - c1d / (2 * sin(2 * trial_angle))
                b1 = 0.5 * c0d
                a0 = 2 * (c0 - a1)
                rotangle = atan(b1 / a1) / 2.0
                cmin = a0 / 2.0 + a1 * cos(2 * rotangle) + b1 * sin(2 * rotangle)
                if c0 < cmin:
                    rotangle += pi / 2.0
                n_min, dummy = rotate_vectors(n_A, rot_unit_A, rotangle)
                self.update_eigenmode(n_min)
                self.update_curvature(cmin)
                self.log(f_rot_A, rotangle)
                if extrapolate:
                    self.forces1E = sin(trial_angle - rotangle) / sin(trial_angle) * self.forces1A + sin(rotangle) / sin(trial_angle) * self.forces1B + (1 - cos(rotangle) - sin(rotangle) * tan(trial_angle / 2.0)) * self.forces0
                else:
                    self.forces1E = None
            if not stoprot:
                if self.control.get_counter('rotcount') >= max_num_rot:
                    stoprot = True
                elif norm(f_rot_A) <= f_rot_max:
                    stoprot = True

    def log(self, f_rot_A, angle):
        """Log each rotational step."""
        if self.logfile is not None:
            if angle:
                l = 'DIM:ROT: %7d %9d %9.4f %9.4f %9.4f\n' % (self.control.get_counter('optcount'), self.control.get_counter('rotcount'), self.get_curvature(), degrees(angle), norm(f_rot_A))
            else:
                l = 'DIM:ROT: %7d %9d %9.4f %9s %9.4f\n' % (self.control.get_counter('optcount'), self.control.get_counter('rotcount'), self.get_curvature(), '---------', norm(f_rot_A))
            self.logfile.write(l)
            self.logfile.flush()

    def get_rotational_force(self):
        """Calculate the rotational force that acts on the dimer."""
        rot_force = perpendicular_vector(self.forces1 - self.forces2, self.eigenmode) / (2.0 * self.dR)
        if self.basis is not None:
            if len(self.basis) == len(self.atoms) and len(self.basis[0]) == 3 and isinstance(self.basis[0][0], float):
                rot_force = perpendicular_vector(rot_force, self.basis)
            else:
                for base in self.basis:
                    rot_force = perpendicular_vector(rot_force, base)
        return rot_force

    def update_curvature(self, curv=None):
        """Update the curvature in the MinModeAtoms object."""
        if curv:
            self.curvature = curv
        else:
            self.curvature = np.vdot(self.forces2 - self.forces1, self.eigenmode) / (2.0 * self.dR)

    def update_eigenmode(self, eigenmode):
        """Update the eigenmode in the MinModeAtoms object."""
        self.eigenmode = eigenmode
        self.update_virtual_positions()
        self.control.increment_counter('rotcount')

    def get_eigenmode(self):
        """Returns the current eigenmode."""
        return self.eigenmode

    def get_curvature(self):
        """Returns the curvature along the current eigenmode."""
        return self.curvature

    def get_control(self):
        """Return the control object."""
        return self.control

    def update_center_forces(self):
        """Get the forces at the center of the dimer."""
        self.atoms.set_positions(self.pos0)
        self.forces0 = self.atoms.get_forces(real=True)
        self.energy0 = self.atoms.get_potential_energy()

    def update_virtual_forces(self, extrapolated_forces=False):
        """Get the forces at the endpoints of the dimer."""
        self.update_virtual_positions()
        if extrapolated_forces:
            self.forces1 = self.forces1E.copy()
        else:
            self.forces1 = self.atoms.get_forces(real=True, pos=self.pos1)
        if self.control.get_parameter('use_central_forces'):
            self.forces2 = 2 * self.forces0 - self.forces1
        else:
            self.forces2 = self.atoms.get_forces(real=True, pos=self.pos2)

    def update_virtual_positions(self):
        """Update the end point positions."""
        self.pos1 = self.pos0 + self.eigenmode * self.dR
        self.pos2 = self.pos0 - self.eigenmode * self.dR

    def set_up_for_eigenmode_search(self):
        """Before eigenmode search, prepare for rotation."""
        self.pos0 = self.atoms.get_positions()
        self.update_center_forces()
        self.update_virtual_positions()
        self.control.reset_counter('rotcount')
        self.forces1E = None

    def set_up_for_optimization_step(self):
        """At the end of rotation, prepare for displacement of the dimer."""
        self.atoms.set_positions(self.pos0)
        self.forces1E = None