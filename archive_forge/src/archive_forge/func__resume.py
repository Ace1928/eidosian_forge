import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
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