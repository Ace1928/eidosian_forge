import os
import warnings
import numpy as np
import ase
from ase.data import atomic_masses
from ase.geometry import cellpar_to_cell
import collections
from functools import reduce
def write_netcdftrajectory(filename, images):
    if hasattr(images, 'get_positions'):
        images = [images]
    with NetCDFTrajectory(filename, mode='w') as traj:
        for atoms in images:
            traj.write(atoms)