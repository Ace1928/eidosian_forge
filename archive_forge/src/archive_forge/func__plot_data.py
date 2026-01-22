import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
def _plot_data(self):
    for step, line in enumerate(self._data):
        self._plot_energy(step, line)
        self._plot_qn(step, line)
        self._plot_md(step, line)
    self._plot_parameters()
    self._ax.set_xlim(self._ax.ax1.get_xlim())