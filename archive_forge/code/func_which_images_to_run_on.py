from ase.io import Trajectory
from ase.io import read
from ase.neb import NEB
from ase.optimize import BFGS
from ase.optimize import FIRE
from ase.calculators.singlepoint import SinglePointCalculator
import ase.parallel as mpi
import numpy as np
import shutil
import os
import types
from math import log
from math import exp
from contextlib import ExitStack
def which_images_to_run_on(self):
    """Determine which set of images to do a NEB at.
        The priority is to first include all images without valid energies,
        secondly include the highest energy image."""
    n_cur = len(self.all_images)
    energies = self.get_energies()
    first_missing = n_cur
    last_missing = 0
    n_missing = 0
    for i in range(1, n_cur - 1):
        if energies[i] != energies[i]:
            n_missing += 1
            first_missing = min(first_missing, i)
            last_missing = max(last_missing, i)
    highest_energy_index = self.get_highest_energy_index()
    nneb = highest_energy_index - 1 - self.n_simul // 2
    nneb = max(nneb, 0)
    nneb = min(nneb, n_cur - self.n_simul - 2)
    nneb = min(nneb, first_missing - 1)
    nneb = max(nneb + self.n_simul, last_missing) - self.n_simul
    to_use = range(nneb, nneb + self.n_simul + 2)
    while self.get_energies_one_image(self.all_images[to_use[0]]) != self.get_energies_one_image(self.all_images[to_use[0]]):
        to_use[0] -= 1
    while self.get_energies_one_image(self.all_images[to_use[-1]]) != self.get_energies_one_image(self.all_images[to_use[-1]]):
        to_use[-1] += 1
    return (to_use, highest_energy_index in to_use[1:-1])