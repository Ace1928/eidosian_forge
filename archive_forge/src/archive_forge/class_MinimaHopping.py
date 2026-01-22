import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
class MinimaHopping:
    """Implements the minima hopping method of global optimization outlined
    by S. Goedecker,  J. Chem. Phys. 120: 9911 (2004). Initialize with an
    ASE atoms object. Optional parameters are fed through keywords.
    To run multiple searches in parallel, specify the minima_traj keyword,
    and have each run point to the same path.
    """
    _default_settings = {'T0': 1000.0, 'beta1': 1.1, 'beta2': 1.1, 'beta3': 1.0 / 1.1, 'Ediff0': 0.5, 'alpha1': 0.98, 'alpha2': 1.0 / 0.98, 'mdmin': 2, 'logfile': 'hop.log', 'minima_threshold': 0.5, 'timestep': 1.0, 'optimizer': QuasiNewton, 'minima_traj': 'minima.traj', 'fmax': 0.05}

    def __init__(self, atoms, **kwargs):
        """Initialize with an ASE atoms object and keyword arguments."""
        self._atoms = atoms
        for key in kwargs:
            if key not in self._default_settings:
                raise RuntimeError('Unknown keyword: %s' % key)
        for k, v in self._default_settings.items():
            setattr(self, '_%s' % k, kwargs.pop(k, v))
        self._passedminimum = PassedMinimum()
        self._previous_optimum = None
        self._previous_energy = None
        self._temperature = self._T0
        self._Ediff = self._Ediff0

    def __call__(self, totalsteps=None, maxtemp=None):
        """Run the minima hopping algorithm. Can specify stopping criteria
        with total steps allowed or maximum searching temperature allowed.
        If neither is specified, runs indefinitely (or until stopped by
        batching software)."""
        self._startup()
        while True:
            if totalsteps and self._counter >= totalsteps:
                self._log('msg', 'Run terminated. Step #%i reached of %i allowed. Increase totalsteps if resuming.' % (self._counter, totalsteps))
                return
            if maxtemp and self._temperature >= maxtemp:
                self._log('msg', 'Run terminated. Temperature is %.2f K; max temperature allowed %.2f K.' % (self._temperature, maxtemp))
                return
            self._previous_optimum = self._atoms.copy()
            self._previous_energy = self._atoms.get_potential_energy()
            self._molecular_dynamics()
            self._optimize()
            self._counter += 1
            self._check_results()

    def _startup(self):
        """Initiates a run, and determines if running from previous data or
        a fresh run."""
        status = np.array(-1.0)
        exists = self._read_minima()
        if world.rank == 0:
            if not exists:
                status = np.array(0.0)
            elif not os.path.exists(self._logfile):
                status = np.array(1.0)
            else:
                status = np.array(2.0)
        world.barrier()
        world.broadcast(status, 0)
        if status == 2.0:
            self._resume()
        else:
            self._counter = 0
            self._log('init')
            self._log('msg', 'Performing initial optimization.')
            if status == 1.0:
                self._log('msg', 'Using existing minima file with %i prior minima: %s' % (len(self._minima), self._minima_traj))
            self._optimize()
            self._check_results()
            self._counter += 1

    def _resume(self):
        """Attempt to resume a run, based on information in the log
        file. Note it will almost always be interrupted in the middle of
        either a qn or md run or when exceeding totalsteps, so it only has
        been tested in those cases currently."""
        f = paropen(self._logfile, 'r')
        lines = f.read().splitlines()
        f.close()
        self._log('msg', 'Attempting to resume stopped run.')
        self._log('msg', 'Using existing minima file with %i prior minima: %s' % (len(self._minima), self._minima_traj))
        mdcount, qncount = (0, 0)
        for line in lines:
            if line[:4] == 'par:' and 'Ediff' not in line:
                self._temperature = float(line.split()[1])
                self._Ediff = float(line.split()[2])
            elif line[:18] == 'msg: Optimization:':
                qncount = int(line[19:].split('qn')[1])
            elif line[:24] == 'msg: Molecular dynamics:':
                mdcount = int(line[25:].split('md')[1])
        self._counter = max((mdcount, qncount))
        if qncount == mdcount:
            self._log('msg', 'Attempting to resume at qn%05i' % qncount)
            if qncount > 0:
                atoms = io.read('qn%05i.traj' % (qncount - 1), index=-1)
                self._previous_optimum = atoms.copy()
                self._previous_energy = atoms.get_potential_energy()
            if os.path.getsize('qn%05i.traj' % qncount) > 0:
                atoms = io.read('qn%05i.traj' % qncount, index=-1)
            else:
                atoms = io.read('md%05i.traj' % qncount, index=-3)
            self._atoms.positions = atoms.get_positions()
            fmax = np.sqrt((atoms.get_forces() ** 2).sum(axis=1).max())
            if fmax < self._fmax:
                self._log('msg', 'qn%05i fmax already less than fmax=%.3f' % (qncount, self._fmax))
                self._counter += 1
                return
            self._optimize()
            self._counter += 1
            if qncount > 0:
                self._check_results()
            else:
                self._record_minimum()
                self._log('msg', 'Found a new minimum.')
                self._log('msg', 'Accepted new minimum.')
                self._log('par')
        elif qncount < mdcount:
            self._log('msg', 'Attempting to resume at md%05i.' % mdcount)
            atoms = io.read('qn%05i.traj' % qncount, index=-1)
            self._previous_optimum = atoms.copy()
            self._previous_energy = atoms.get_potential_energy()
            self._molecular_dynamics(resume=mdcount)
            self._optimize()
            self._counter += 1
            self._check_results()

    def _check_results(self):
        """Adjusts parameters and positions based on outputs."""
        self._read_minima()
        if len(self._minima) == 0:
            self._log('msg', 'Found a new minimum.')
            self._log('msg', 'Accepted new minimum.')
            self._record_minimum()
            self._log('par')
            return
        if self._previous_optimum:
            compare = ComparePositions(translate=False)
            dmax = compare(self._atoms, self._previous_optimum)
            self._log('msg', 'Max distance to last minimum: %.3f A' % dmax)
            if dmax < self._minima_threshold:
                self._log('msg', 'Re-found last minimum.')
                self._temperature *= self._beta1
                self._log('par')
                return
        unique, dmax_closest = self._unique_minimum_position()
        self._log('msg', 'Max distance to closest minimum: %.3f A' % dmax_closest)
        if not unique:
            self._temperature *= self._beta2
            self._log('msg', 'Found previously found minimum.')
            self._log('par')
            if self._previous_optimum:
                self._log('msg', 'Restoring last minimum.')
                self._atoms.positions = self._previous_optimum.positions
            return
        self._temperature *= self._beta3
        self._log('msg', 'Found a new minimum.')
        self._log('par')
        if self._previous_energy is None or self._atoms.get_potential_energy() < self._previous_energy + self._Ediff:
            self._log('msg', 'Accepted new minimum.')
            self._Ediff *= self._alpha1
            self._log('par')
            self._record_minimum()
        else:
            self._log('msg', 'Rejected new minimum due to energy. Restoring last minimum.')
            self._atoms.positions = self._previous_optimum.positions
            self._Ediff *= self._alpha2
            self._log('par')

    def _log(self, cat='msg', message=None):
        """Records the message as a line in the log file."""
        if cat == 'init':
            if world.rank == 0:
                if os.path.exists(self._logfile):
                    raise RuntimeError('File exists: %s' % self._logfile)
            fd = paropen(self._logfile, 'w')
            fd.write('par: %12s %12s %12s\n' % ('T (K)', 'Ediff (eV)', 'mdmin'))
            fd.write('ene: %12s %12s %12s\n' % ('E_current', 'E_previous', 'Difference'))
            fd.close()
            return
        fd = paropen(self._logfile, 'a')
        if cat == 'msg':
            line = 'msg: %s' % message
        elif cat == 'par':
            line = 'par: %12.4f %12.4f %12i' % (self._temperature, self._Ediff, self._mdmin)
        elif cat == 'ene':
            current = self._atoms.get_potential_energy()
            if self._previous_optimum:
                previous = self._previous_energy
                line = 'ene: %12.5f %12.5f %12.5f' % (current, previous, current - previous)
            else:
                line = 'ene: %12.5f' % current
        fd.write(line + '\n')
        fd.close()

    def _optimize(self):
        """Perform an optimization."""
        self._atoms.set_momenta(np.zeros(self._atoms.get_momenta().shape))
        with self._optimizer(self._atoms, trajectory='qn%05i.traj' % self._counter, logfile='qn%05i.log' % self._counter) as opt:
            self._log('msg', 'Optimization: qn%05i' % self._counter)
            opt.run(fmax=self._fmax)
            self._log('ene')

    def _record_minimum(self):
        """Adds the current atoms configuration to the minima list."""
        with io.Trajectory(self._minima_traj, 'a') as traj:
            traj.write(self._atoms)
        self._read_minima()
        self._log('msg', 'Recorded minima #%i.' % (len(self._minima) - 1))

    def _read_minima(self):
        """Reads in the list of minima from the minima file."""
        exists = os.path.exists(self._minima_traj)
        if exists:
            empty = os.path.getsize(self._minima_traj) == 0
            if not empty:
                with io.Trajectory(self._minima_traj, 'r') as traj:
                    self._minima = [atoms for atoms in traj]
            else:
                self._minima = []
            return True
        else:
            self._minima = []
            return False

    def _molecular_dynamics(self, resume=None):
        """Performs a molecular dynamics simulation, until mdmin is
        exceeded. If resuming, the file number (md%05i) is expected."""
        self._log('msg', 'Molecular dynamics: md%05i' % self._counter)
        mincount = 0
        energies, oldpositions = ([], [])
        thermalized = False
        if resume:
            self._log('msg', 'Resuming MD from md%05i.traj' % resume)
            if os.path.getsize('md%05i.traj' % resume) == 0:
                self._log('msg', 'md%05i.traj is empty. Resuming from qn%05i.traj.' % (resume, resume - 1))
                atoms = io.read('qn%05i.traj' % (resume - 1), index=-1)
            else:
                with io.Trajectory('md%05i.traj' % resume, 'r') as images:
                    for atoms in images:
                        energies.append(atoms.get_potential_energy())
                        oldpositions.append(atoms.positions.copy())
                        passedmin = self._passedminimum(energies)
                        if passedmin:
                            mincount += 1
                self._atoms.set_momenta(atoms.get_momenta())
                thermalized = True
            self._atoms.positions = atoms.get_positions()
            self._log('msg', 'Starting MD with %i existing energies.' % len(energies))
        if not thermalized:
            MaxwellBoltzmannDistribution(self._atoms, temperature_K=self._temperature, force_temp=True)
        traj = io.Trajectory('md%05i.traj' % self._counter, 'a', self._atoms)
        dyn = VelocityVerlet(self._atoms, timestep=self._timestep * units.fs)
        log = MDLogger(dyn, self._atoms, 'md%05i.log' % self._counter, header=True, stress=False, peratom=False)
        with traj, dyn, log:
            dyn.attach(log, interval=1)
            dyn.attach(traj, interval=1)
            while mincount < self._mdmin:
                dyn.run(1)
                energies.append(self._atoms.get_potential_energy())
                passedmin = self._passedminimum(energies)
                if passedmin:
                    mincount += 1
                oldpositions.append(self._atoms.positions.copy())
            self._atoms.positions = oldpositions[passedmin[0]]

    def _unique_minimum_position(self):
        """Identifies if the current position of the atoms, which should be
        a local minima, has been found before."""
        unique = True
        dmax_closest = 99999.0
        compare = ComparePositions(translate=True)
        self._read_minima()
        for minimum in self._minima:
            dmax = compare(minimum, self._atoms)
            if dmax < self._minima_threshold:
                unique = False
            if dmax < dmax_closest:
                dmax_closest = dmax
        return (unique, dmax_closest)