from __future__ import annotations
import os
import tempfile
from shutil import which
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import deprecated
from monty.tempfile import ScratchDir
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.io.packmol import PackmolBoxGen
from pymatgen.util.coord import get_angle
class LammpsRunner:
    """LAMMPS wrapper."""

    def __init__(self, input_filename: str='lammps.in', bin: str='lammps') -> None:
        """
        Args:
            input_filename (str): input file name
            bin (str): command to run, excluding the input file name.
        """
        self.lammps_bin = bin.split()
        if not which(self.lammps_bin[-1]):
            raise RuntimeError(f"LammpsRunner requires the executable {self.lammps_bin[-1]} to be in the path. Please download and install LAMMPS from https://www.lammps.org/. Don't forget to add the binary to your path")
        self.input_filename = input_filename

    def run(self) -> tuple[bytes, bytes]:
        """Write the input/data files and run LAMMPS."""
        lammps_cmd = [*self.lammps_bin, '-in', self.input_filename]
        print(f'Running: {' '.join(lammps_cmd)}')
        with Popen(lammps_cmd, stdout=PIPE, stderr=PIPE) as p:
            stdout, stderr = p.communicate()
        return (stdout, stderr)