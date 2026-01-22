import numpy as np
import scipy.optimize as opt
from ase.optimize.optimize import Optimizer
class SciPyOptimizer(Optimizer):
    """General interface for SciPy optimizers

    Only the call to the optimizer is still needed
    """

    def __init__(self, atoms, logfile='-', trajectory=None, callback_always=False, alpha=70.0, master=None, force_consistent=None):
        """Initialize object

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        callback_always: book
            Should the callback be run after each force call (also in the
            linesearch)

        alpha: float
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K).  By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.
        """
        restart = None
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master, force_consistent=force_consistent)
        self.force_calls = 0
        self.callback_always = callback_always
        self.H0 = alpha

    def x0(self):
        """Return x0 in a way SciPy can use

        This class is mostly usable for subclasses wanting to redefine the
        parameters (and the objective function)"""
        return self.atoms.get_positions().reshape(-1)

    def f(self, x):
        """Objective function for use of the optimizers"""
        self.atoms.set_positions(x.reshape(-1, 3))
        return self.atoms.get_potential_energy(force_consistent=self.force_consistent) / self.H0

    def fprime(self, x):
        """Gradient of the objective function for use of the optimizers"""
        self.atoms.set_positions(x.reshape(-1, 3))
        self.force_calls += 1
        if self.callback_always:
            self.callback(x)
        return -self.atoms.get_forces().reshape(-1) / self.H0

    def callback(self, x):
        """Callback function to be run after each iteration by SciPy

        This should also be called once before optimization starts, as SciPy
        optimizers only calls it after each iteration, while ase optimizers
        call something similar before as well.
        
        :meth:`callback`() can raise a :exc:`Converged` exception to signal the
        optimisation is complete. This will be silently ignored by
        :meth:`run`().
        """
        f = self.atoms.get_forces()
        self.log(f)
        self.call_observers()
        if self.converged(f):
            raise Converged
        self.nsteps += 1

    def run(self, fmax=0.05, steps=100000000):
        if self.force_consistent is None:
            self.set_force_consistent()
        self.fmax = fmax
        try:
            self.callback(None)
            self.call_fmin(fmax / self.H0, steps)
        except Converged:
            pass

    def dump(self, data):
        pass

    def load(self):
        pass

    def call_fmin(self, fmax, steps):
        raise NotImplementedError