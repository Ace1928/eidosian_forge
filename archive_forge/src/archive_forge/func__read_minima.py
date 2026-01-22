import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
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