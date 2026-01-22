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
class MinModeAtoms:
    """Wrapper for Atoms with information related to minimum mode searching.

    Contains an Atoms object and pipes all unknown function calls to that
    object.
    Other information that is stored in this object are the estimate for
    the lowest eigenvalue, *curvature*, and its corresponding eigenmode,
    *eigenmode*. Furthermore, the original configuration of the Atoms
    object is stored for use in multiple minimum mode searches.
    The forces on the system are modified by inverting the component
    along the eigenmode estimate. This eventually brings the system to
    a saddle point.

    Parameters:

    atoms : Atoms object
        A regular Atoms object
    control : MinModeControl object
        Contains the parameters necessary for the eigenmode search.
        If no control object is supplied a default DimerControl
        will be created and used.
    mask: list of bool
        Determines which atoms will be moved when calling displace()
    random_seed: int
        The seed used for the random number generator. Defaults to
        modified version the current time.

    References: [1]_ [2]_ [3]_ [4]_

    .. [1] Henkelman and Jonsson, JCP 111, 7010 (1999)
    .. [2] Olsen, Kroes, Henkelman, Arnaldsson, and Jonsson, JCP 121,
           9776 (2004).
    .. [3] Heyden, Bell, and Keil, JCP 123, 224101 (2005).
    .. [4] Kastner and Sherwood, JCP 128, 014106 (2008).

    """

    def __init__(self, atoms, control=None, eigenmodes=None, random_seed=None, **kwargs):
        self.minmode_init = True
        self.atoms = atoms
        self.eigenmodes = eigenmodes
        self.curvatures = None
        if control is None:
            self.control = DimerControl(**kwargs)
            w = 'Missing control object in ' + self.__class__.__name__ + '. Using default: DimerControl()'
            warnings.warn(w, UserWarning)
            if self.control.logfile is not None:
                self.control.logfile.write('DIM:WARN: ' + w + '\n')
                self.control.logfile.flush()
        else:
            self.control = control
            logfile = self.control.get_logfile()
            mlogfile = self.control.get_eigenmode_logfile()
            for key in kwargs:
                if key == 'logfile':
                    logfile = kwargs[key]
                elif key == 'eigenmode_logfile':
                    mlogfile = kwargs[key]
                else:
                    self.control.set_parameter(key, kwargs[key])
            self.control.initialize_logfiles(logfile=logfile, eigenmode_logfile=mlogfile)
        if random_seed is None:
            t = time.time()
            if world.size > 1:
                t = world.sum(t) / world.size
            random_seed = int(('%30.9f' % t)[-9:])
        self.random_state = np.random.RandomState(random_seed)
        self.order = self.control.get_parameter('order')
        self.curvatures = [100.0] * self.order
        self.atoms0 = self.atoms.copy()
        self.save_original_forces()
        self.logfile = self.control.get_logfile()
        self.mlogfile = self.control.get_eigenmode_logfile()

    def save_original_forces(self, force_calculation=False):
        """Store the forces (and energy) of the original state."""
        if self.calc is not None:
            if hasattr(self.calc, 'calculation_required') and (not self.calc.calculation_required(self.atoms, ['energy', 'forces'])) or force_calculation:
                calc = SinglePointCalculator(self.atoms0, energy=self.atoms.get_potential_energy(), forces=self.atoms.get_forces())
                self.atoms0.calc = calc

    def initialize_eigenmodes(self, method=None, eigenmodes=None, gauss_std=None):
        """Make an initial guess for the eigenmode."""
        if eigenmodes is None:
            pos = self.get_positions()
            old_pos = self.get_original_positions()
            if method == None:
                method = self.control.get_parameter('initial_eigenmode_method')
            if method.lower() == 'displacement' and (pos - old_pos).any():
                eigenmode = normalize(pos - old_pos)
            elif method.lower() == 'gauss':
                self.displace(log=False, gauss_std=gauss_std, method=method)
                new_pos = self.get_positions()
                eigenmode = normalize(new_pos - pos)
                self.set_positions(pos)
            else:
                e = "initial_eigenmode must use either 'gauss' or " + "'displacement', if the latter is used the atoms " + 'must have moved away from the original positions.' + "You have requested '%s'." % method
                raise NotImplementedError(e)
            eigenmodes = [eigenmode]
        if self.order > 1:
            if len(eigenmodes) == 1:
                for k in range(1, self.order):
                    pos = self.get_positions()
                    self.displace(log=False, gauss_std=gauss_std, method=method)
                    new_pos = self.get_positions()
                    eigenmode = normalize(new_pos - pos)
                    self.set_positions(pos)
                    eigenmodes += [eigenmode]
        self.eigenmodes = eigenmodes
        if self.order > 1:
            for k in range(self.order):
                self.ensure_eigenmode_orthogonality(k)
        self.eigenmode_log()

    def calculation_required(self):
        """Check if a calculation is required."""
        return self.minmode_init or self.check_atoms != self.atoms

    def calculate_real_forces_and_energies(self, **kwargs):
        """Calculate and store the potential energy and forces."""
        if self.minmode_init:
            self.minmode_init = False
            self.initialize_eigenmodes(eigenmodes=self.eigenmodes)
        self.rotation_required = True
        self.forces0 = self.atoms.get_forces(**kwargs)
        self.energy0 = self.atoms.get_potential_energy()
        self.control.increment_counter('forcecalls')
        self.check_atoms = self.atoms.copy()

    def get_potential_energy(self):
        """Return the potential energy."""
        if self.calculation_required():
            self.calculate_real_forces_and_energies()
        return self.energy0

    def get_forces(self, real=False, pos=None, **kwargs):
        """Return the forces, projected or real."""
        if self.calculation_required() and pos is None:
            self.calculate_real_forces_and_energies(**kwargs)
        if real and pos is None:
            return self.forces0
        elif real and pos is not None:
            old_pos = self.atoms.get_positions()
            self.atoms.set_positions(pos)
            forces = self.atoms.get_forces()
            self.control.increment_counter('forcecalls')
            self.atoms.set_positions(old_pos)
            return forces
        else:
            if self.rotation_required:
                self.find_eigenmodes(order=self.order)
                self.eigenmode_log()
                self.rotation_required = False
                self.control.increment_counter('optcount')
            return self.get_projected_forces()

    def ensure_eigenmode_orthogonality(self, order):
        mode = self.eigenmodes[order - 1].copy()
        for k in range(order - 1):
            mode = perpendicular_vector(mode, self.eigenmodes[k])
        self.eigenmodes[order - 1] = normalize(mode)

    def find_eigenmodes(self, order=1):
        """Launch eigenmode searches."""
        if self.control.get_parameter('eigenmode_method').lower() != 'dimer':
            e = 'Only the Dimer control object has been implemented.'
            raise NotImplementedError(e)
        for k in range(order):
            if k > 0:
                self.ensure_eigenmode_orthogonality(k + 1)
            search = DimerEigenmodeSearch(self, self.control, eigenmode=self.eigenmodes[k], basis=self.eigenmodes[:k])
            search.converge_to_eigenmode()
            search.set_up_for_optimization_step()
            self.eigenmodes[k] = search.get_eigenmode()
            self.curvatures[k] = search.get_curvature()

    def get_projected_forces(self, pos=None):
        """Return the projected forces."""
        if pos is not None:
            forces = self.get_forces(real=True, pos=pos).copy()
        else:
            forces = self.forces0.copy()
        for k, mode in enumerate(self.eigenmodes):
            if self.get_curvature(order=k) > 0.0 and self.order == 1:
                forces = -parallel_vector(forces, mode)
            else:
                forces -= 2 * parallel_vector(forces, mode)
        return forces

    def restore_original_positions(self):
        """Restore the MinModeAtoms object positions to the original state."""
        self.atoms.set_positions(self.get_original_positions())

    def get_barrier_energy(self):
        """The energy difference between the current and original states"""
        try:
            original_energy = self.get_original_potential_energy()
            dimer_energy = self.get_potential_energy()
            return dimer_energy - original_energy
        except RuntimeError:
            w = 'The potential energy is not available, without further ' + 'calculations, most likely at the original state.'
            warnings.warn(w, UserWarning)
            return np.nan

    def get_control(self):
        """Return the control object."""
        return self.control

    def get_curvature(self, order='max'):
        """Return the eigenvalue estimate."""
        if order == 'max':
            return max(self.curvatures)
        else:
            return self.curvatures[order - 1]

    def get_eigenmode(self, order=1):
        """Return the current eigenmode guess."""
        return self.eigenmodes[order - 1]

    def get_atoms(self):
        """Return the unextended Atoms object."""
        return self.atoms

    def set_atoms(self, atoms):
        """Set a new Atoms object"""
        self.atoms = atoms

    def set_eigenmode(self, eigenmode, order=1):
        """Set the eigenmode guess."""
        self.eigenmodes[order - 1] = eigenmode

    def set_curvature(self, curvature, order=1):
        """Set the eigenvalue estimate."""
        self.curvatures[order - 1] = curvature

    def __getattr__(self, attr):
        """Return any value of the Atoms object"""
        if 'original' in attr.split('_'):
            attr = attr.replace('_original_', '_')
            return getattr(self.atoms0, attr)
        else:
            return getattr(self.atoms, attr)

    def __len__(self):
        return len(self.atoms)

    def displace(self, displacement_vector=None, mask=None, method=None, displacement_center=None, radius=None, number_of_atoms=None, gauss_std=None, mic=True, log=True):
        """Move the atoms away from their current position.

        This is one of the essential parts of minimum mode searches.
        The parameters can all be set in the control object and overwritten
        when this method is run, apart from *displacement_vector*.
        It is preferred to modify the control values rather than those here
        in order for the correct ones to show up in the log file.

        *method* can be either 'gauss' for random displacement or 'vector'
        to perform a predefined displacement.

        *gauss_std* is the standard deviation of the gauss curve that is
        used for random displacement.

        *displacement_center* can be either the number of an atom or a 3D
        position. It must be accompanied by a *radius* (all atoms within it
        will be displaced) or a *number_of_atoms* which decides how many of
        the closest atoms will be displaced.

        *mic* controls the usage of the Minimum Image Convention.

        If both *mask* and *displacement_center* are used, the atoms marked
        as False in the *mask* will not be affected even though they are
        within reach of the *displacement_center*.

        The parameters priority order:
        1) displacement_vector
        2) mask
        3) displacement_center (with radius and/or number_of_atoms)

        If both *radius* and *number_of_atoms* are supplied with
        *displacement_center*, only atoms that fulfill both criteria will
        be displaced.

        """
        if mask is None:
            mask = self.control.get_parameter('mask')
        if method is None:
            method = self.control.get_parameter('displacement_method')
        if gauss_std is None:
            gauss_std = self.control.get_parameter('gauss_std')
        if displacement_center is None:
            displacement_center = self.control.get_parameter('displacement_center')
        if radius is None:
            radius = self.control.get_parameter('displacement_radius')
        if number_of_atoms is None:
            number_of_atoms = self.control.get_parameter('number_of_displacement_atoms')
        if displacement_vector is not None and method.lower() != 'vector':
            e = 'displacement_vector was supplied but a different method ' + "('%s') was chosen.\n" % str(method)
            raise ValueError(e)
        elif displacement_vector is None and method.lower() == 'vector':
            e = 'A displacement_vector must be supplied when using ' + "method = '%s'.\n" % str(method)
            raise ValueError(e)
        elif displacement_center is not None and radius is None and (number_of_atoms is None):
            e = 'When displacement_center is chosen, either radius or ' + 'number_of_atoms must be supplied.\n'
            raise ValueError(e)
        if displacement_center is not None:
            c = displacement_center
            if isinstance(c, int):
                c = displacement_center % len(self)
                d = [(k, self.get_distance(k, c, mic=mic)) for k in range(len(self))]
            elif len(c) == 3 and [type(c_k) for c_k in c] == [float] * 3:
                d = [(k, norm(self.get_positions()[k] - c)) for k in range(len(self))]
            else:
                e = 'displacement_center must be either the number of an ' + 'atom in MinModeAtoms object or a 3D position ' + '(3-tuple of floats).'
                raise ValueError(e)
            if radius is not None:
                r_mask = [dist[1] < radius for dist in d]
            else:
                r_mask = [True for _ in range(len(self))]
            if number_of_atoms is not None:
                d_sorted = [n[0] for n in sorted(d, key=lambda k: k[1])]
                n_nearest = d_sorted[:number_of_atoms]
                n_mask = [k in n_nearest for k in range(len(self))]
            else:
                n_mask = [True for _ in range(len(self))]
            c_mask = [n_mask[k] and r_mask[k] for k in range(len(self))]
        else:
            c_mask = None
        if mask is None:
            mask = [True for _ in range(len(self))]
            if c_mask is None:
                w = 'It was not possible to figure out which atoms to ' + 'displace, Will try to displace all atoms.\n'
                warnings.warn(w, UserWarning)
                if self.logfile is not None:
                    self.logfile.write('MINMODE:WARN: ' + w + '\n')
                    self.logfile.flush()
        if c_mask is not None:
            mask = [mask[k] and c_mask[k] for k in range(len(self))]
        if displacement_vector is None:
            displacement_vector = []
            for k in range(len(self)):
                if mask[k]:
                    diff_line = []
                    for _ in range(3):
                        if method.lower() == 'gauss':
                            if not gauss_std:
                                gauss_std = self.control.get_parameter('gauss_std')
                            diff = self.random_state.normal(0.0, gauss_std)
                        else:
                            e = 'Invalid displacement method >>%s<<' % str(method)
                            raise ValueError(e)
                        diff_line.append(diff)
                    displacement_vector.append(diff_line)
                else:
                    displacement_vector.append([0.0] * 3)
        for k in range(len(mask)):
            if not mask[k]:
                displacement_vector[k] = [0.0] * 3
        if log:
            pos0 = self.get_positions()
        self.set_positions(self.get_positions() + displacement_vector)
        if log:
            parameters = {'mask': mask, 'displacement_method': method, 'gauss_std': gauss_std, 'displacement_center': displacement_center, 'displacement_radius': radius, 'number_of_displacement_atoms': number_of_atoms}
            self.displacement_log(self.get_positions() - pos0, parameters)

    def eigenmode_log(self):
        """Log the eigenmodes (eigenmode estimates)"""
        if self.mlogfile is not None:
            l = 'MINMODE:MODE: Optimization Step: %i\n' % self.control.get_counter('optcount')
            for m_num, mode in enumerate(self.eigenmodes):
                l += 'MINMODE:MODE: Order: %i\n' % m_num
                for k in range(len(mode)):
                    l += 'MINMODE:MODE: %7i %15.8f %15.8f %15.8f\n' % (k, mode[k][0], mode[k][1], mode[k][2])
            self.mlogfile.write(l)
            self.mlogfile.flush()

    def displacement_log(self, displacement_vector, parameters):
        """Log the displacement"""
        if self.logfile is not None:
            lp = 'MINMODE:DISP: Parameters, different from the control:\n'
            mod_para = False
            for key in parameters:
                if parameters[key] != self.control.get_parameter(key):
                    lp += 'MINMODE:DISP: %s = %s\n' % (str(key), str(parameters[key]))
                    mod_para = True
            if mod_para:
                l = lp
            else:
                l = ''
            for k in range(len(displacement_vector)):
                l += 'MINMODE:DISP: %7i %15.8f %15.8f %15.8f\n' % (k, displacement_vector[k][0], displacement_vector[k][1], displacement_vector[k][2])
            self.logfile.write(l)
            self.logfile.flush()

    def summarize(self):
        """Summarize the Minimum mode search."""
        if self.logfile is None:
            logfile = sys.stdout
        else:
            logfile = self.logfile
        c = self.control
        label = 'MINMODE:SUMMARY: '
        l = label + '-------------------------\n'
        l += label + 'Barrier: %16.4f\n' % self.get_barrier_energy()
        l += label + 'Curvature: %14.4f\n' % self.get_curvature()
        l += label + 'Optimizer steps: %8i\n' % c.get_counter('optcount')
        l += label + 'Forcecalls: %13i\n' % c.get_counter('forcecalls')
        l += label + '-------------------------\n'
        logfile.write(l)