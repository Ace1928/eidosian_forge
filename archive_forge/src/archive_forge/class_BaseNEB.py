import sys
import threading
import warnings
from abc import ABC, abstractmethod
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumtrapz
import ase.parallel
from ase.build import minimize_rotation_and_translation
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import MDMin
from ase.optimize.optimize import Optimizer
from ase.optimize.sciopt import OptimizerConvergenceError
from ase.geometry import find_mic
from ase.utils import lazyproperty, deprecated
from ase.utils.forcecurve import fit_images
from ase.optimize.precon import Precon, PreconImages
from ase.optimize.ode import ode12r
class BaseNEB:

    def __init__(self, images, k=0.1, climb=False, parallel=False, remove_rotation_and_translation=False, world=None, method='aseneb', allow_shared_calculator=False, precon=None):
        self.images = images
        self.climb = climb
        self.parallel = parallel
        self.allow_shared_calculator = allow_shared_calculator
        for img in images:
            if len(img) != self.natoms:
                raise ValueError('Images have different numbers of atoms')
            if np.any(img.pbc != images[0].pbc):
                raise ValueError('Images have different boundary conditions')
            if np.any(img.get_atomic_numbers() != images[0].get_atomic_numbers()):
                raise ValueError('Images have atoms in different orders')
            if np.any(np.abs(img.get_cell() - images[0].get_cell()) > 1e-08):
                raise NotImplementedError('Variable cell NEB is not implemented yet')
        self.emax = np.nan
        self.remove_rotation_and_translation = remove_rotation_and_translation
        if method in ['aseneb', 'eb', 'improvedtangent', 'spline', 'string']:
            self.method = method
        else:
            raise NotImplementedError(method)
        if precon is not None and method not in ['spline', 'string']:
            raise NotImplementedError(f'no precon implemented: {method}')
        self.precon = precon
        self.neb_method = get_neb_method(self, method)
        if isinstance(k, (float, int)):
            k = [k] * (self.nimages - 1)
        self.k = list(k)
        if world is None:
            world = ase.parallel.world
        self.world = world
        if parallel:
            if self.allow_shared_calculator:
                raise RuntimeError('Cannot use shared calculators in parallel in NEB.')
        self.real_forces = None
        self.energies = None
        self.residuals = None

    @property
    def natoms(self):
        return len(self.images[0])

    @property
    def nimages(self):
        return len(self.images)

    @staticmethod
    def freeze_results_on_image(atoms: ase.Atoms, **results_to_include):
        atoms.calc = SinglePointCalculator(atoms=atoms, **results_to_include)

    def interpolate(self, method='linear', mic=False, apply_constraint=None):
        """Interpolate the positions of the interior images between the
        initial state (image 0) and final state (image -1).

        method: str
            Method by which to interpolate: 'linear' or 'idpp'.
            linear provides a standard straight-line interpolation, while
            idpp uses an image-dependent pair potential.
        mic: bool
            Use the minimum-image convention when interpolating.
        apply_constraint: bool
            Controls if the constraints attached to the images
            are ignored or applied when setting the interpolated positions.
            Default value is None, in this case the resulting constrained
            positions (apply_constraint=True) are compared with unconstrained
            positions (apply_constraint=False),
            if the positions are not the same
            the user is required to specify the desired behaviour
            by setting up apply_constraint keyword argument to False or True.
        """
        if self.remove_rotation_and_translation:
            minimize_rotation_and_translation(self.images[0], self.images[-1])
        interpolate(self.images, mic, apply_constraint=apply_constraint)
        if method == 'idpp':
            idpp_interpolate(images=self, traj=None, log=None, mic=mic)

    @deprecated("Please use NEB's interpolate(method='idpp') method or directly call the idpp_interpolate function from ase.neb")
    def idpp_interpolate(self, traj='idpp.traj', log='idpp.log', fmax=0.1, optimizer=MDMin, mic=False, steps=100):
        idpp_interpolate(self, traj=traj, log=log, fmax=fmax, optimizer=optimizer, mic=mic, steps=steps)

    def get_positions(self):
        positions = np.empty(((self.nimages - 2) * self.natoms, 3))
        n1 = 0
        for image in self.images[1:-1]:
            n2 = n1 + self.natoms
            positions[n1:n2] = image.get_positions()
            n1 = n2
        return positions

    def set_positions(self, positions, adjust_positions=True):
        if adjust_positions:
            positions = self.neb_method.adjust_positions(positions)
        n1 = 0
        for image in self.images[1:-1]:
            n2 = n1 + self.natoms
            image.set_positions(positions[n1:n2])
            n1 = n2

    def get_forces(self):
        """Evaluate and return the forces."""
        images = self.images
        if not self.allow_shared_calculator:
            calculators = [image.calc for image in images if image.calc is not None]
            if len(set(calculators)) != len(calculators):
                msg = 'One or more NEB images share the same calculator.  Each image must have its own calculator.  You may wish to use the ase.neb.SingleCalculatorNEB class instead, although using separate calculators is recommended.'
                raise ValueError(msg)
        forces = np.empty((self.nimages - 2, self.natoms, 3))
        energies = np.empty(self.nimages)
        if self.remove_rotation_and_translation:
            for i in range(1, self.nimages):
                minimize_rotation_and_translation(images[i - 1], images[i])
        if self.method != 'aseneb':
            energies[0] = images[0].get_potential_energy()
            energies[-1] = images[-1].get_potential_energy()
        if not self.parallel:
            for i in range(1, self.nimages - 1):
                energies[i] = images[i].get_potential_energy()
                forces[i - 1] = images[i].get_forces()
        elif self.world.size == 1:

            def run(image, energies, forces):
                energies[:] = image.get_potential_energy()
                forces[:] = image.get_forces()
            threads = [threading.Thread(target=run, args=(images[i], energies[i:i + 1], forces[i - 1:i])) for i in range(1, self.nimages - 1)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            i = self.world.rank * (self.nimages - 2) // self.world.size + 1
            try:
                energies[i] = images[i].get_potential_energy()
                forces[i - 1] = images[i].get_forces()
            except Exception:
                error = self.world.sum(1.0)
                raise
            else:
                error = self.world.sum(0.0)
                if error:
                    raise RuntimeError('Parallel NEB failed!')
            for i in range(1, self.nimages - 1):
                root = (i - 1) * self.world.size // (self.nimages - 2)
                self.world.broadcast(energies[i:i + 1], root)
                self.world.broadcast(forces[i - 1], root)
        if self.precon is None or isinstance(self.precon, str) or isinstance(self.precon, Precon):
            self.precon = PreconImages(self.precon, images)
        precon_forces = self.precon.apply(forces, index=slice(1, -1))
        self.energies = energies
        self.real_forces = np.zeros((self.nimages, self.natoms, 3))
        self.real_forces[1:-1] = forces
        state = NEBState(self, images, energies)
        self.imax = state.imax
        self.emax = state.emax
        spring1 = state.spring(0)
        self.residuals = []
        for i in range(1, self.nimages - 1):
            spring2 = state.spring(i)
            tangent = self.neb_method.get_tangent(state, spring1, spring2, i)
            tangential_force = np.vdot(forces[i - 1], tangent)
            imgforce = precon_forces[i - 1]
            if i == self.imax and self.climb:
                'The climbing image, imax, is not affected by the spring\n                   forces. This image feels the full PES-derived force,\n                   but the tangential component is inverted:\n                   see Eq. 5 in paper II.'
                if self.method == 'aseneb':
                    tangent_mag = np.vdot(tangent, tangent)
                    imgforce -= 2 * tangential_force / tangent_mag * tangent
                else:
                    imgforce -= 2 * tangential_force * tangent
            else:
                self.neb_method.add_image_force(state, tangential_force, tangent, imgforce, spring1, spring2, i)
                residual = self.precon.get_residual(i, imgforce)
                self.residuals.append(residual)
            spring1 = spring2
        return precon_forces.reshape((-1, 3))

    def get_residual(self):
        """Return residual force along the band.

        Typically this the maximum force component on any image. For
        non-trivial preconditioners, the appropriate preconditioned norm
        is used to compute the residual.
        """
        if self.residuals is None:
            raise RuntimeError('get_residual() called before get_forces()')
        return np.max(self.residuals)

    def get_potential_energy(self, force_consistent=False):
        """Return the maximum potential energy along the band.
        Note that the force_consistent keyword is ignored and is only
        present for compatibility with ase.Atoms.get_potential_energy."""
        return self.emax

    def set_calculators(self, calculators):
        """Set new calculators to the images.

        Parameters
        ----------
        calculators : Calculator / list(Calculator)
            calculator(s) to attach to images
              - single calculator, only if allow_shared_calculator=True
            list of calculators if length:
              - length nimages, set to all images
              - length nimages-2, set to non-end images only
        """
        if not isinstance(calculators, list):
            if self.allow_shared_calculator:
                calculators = [calculators] * self.nimages
            else:
                raise RuntimeError('Cannot set shared calculator to NEB with allow_shared_calculator=False')
        n = len(calculators)
        if n == self.nimages:
            for i in range(self.nimages):
                self.images[i].calc = calculators[i]
        elif n == self.nimages - 2:
            for i in range(1, self.nimages - 1):
                self.images[i].calc = calculators[i - 1]
        else:
            raise RuntimeError('len(calculators)=%d does not fit to len(images)=%d' % (n, self.nimages))

    def __len__(self):
        return (self.nimages - 2) * self.natoms

    def iterimages(self):
        for i, atoms in enumerate(self.images):
            if i == 0 or i == self.nimages - 1:
                yield atoms
            else:
                atoms = atoms.copy()
                self.freeze_results_on_image(atoms, energy=self.energies[i], forces=self.real_forces[i])
                yield atoms

    def spline_fit(self, positions=None, norm='precon'):
        """
        Fit a cubic spline to this NEB

        Args:
            norm (str, optional): Norm to use: 'precon' (default) or 'euclidean'

        Returns:
            fit: ase.precon.precon.SplineFit instance
        """
        if norm == 'precon':
            if self.precon is None or isinstance(self.precon, str):
                self.precon = PreconImages(self.precon, self.images)
            precon = self.precon
        elif norm == 'euclidean':
            precon = PreconImages('ID', self.images)
        else:
            raise ValueError(f'unsupported norm {norm}')
        return precon.spline_fit(positions)

    def integrate_forces(self, spline_points=1000, bc_type='not-a-knot'):
        """Use spline fit to integrate forces along MEP to approximate
        energy differences using the virtual work approach.

        Args:
            spline_points (int, optional): Number of points. Defaults to 1000.
            bc_type (str, optional): Boundary conditions, default 'not-a-knot'.

        Returns:
            s: reaction coordinate in range [0, 1], with `spline_points` entries
            E: result of integrating forces, on the same grid as `s`.
            F: projected forces along MEP
        """
        fit = self.spline_fit(norm='euclidean')
        forces = np.array([image.get_forces().reshape(-1) for image in self.images])
        f = CubicSpline(fit.s, forces, bc_type=bc_type)
        s = np.linspace(0.0, 1.0, spline_points, endpoint=True)
        dE = f(s) * fit.dx_ds(s)
        F = dE.sum(axis=1)
        E = -cumtrapz(F, s, initial=0.0)
        return (s, E, F)