from __future__ import annotations
import os
import re
import shutil
import subprocess
from string import Template
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Molecule
def _gw_run(self):
    """Performs FIESTA (gw) run."""
    if self.folder != os.getcwd():
        init_folder = os.getcwd()
        os.chdir(self.folder)
    with zopen(self.log_file, mode='w') as fout:
        subprocess.call(['mpirun', '-n', str(self.mpi_procs), 'fiesta', str(self.grid[0]), str(self.grid[1]), str(self.grid[2])], stdout=fout)
    if self.folder != os.getcwd():
        os.chdir(init_folder)