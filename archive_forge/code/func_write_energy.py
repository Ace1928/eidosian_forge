from __future__ import annotations
import logging
import math
import os
import subprocess
import tempfile
import time
from shutil import which
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.dev import requires
from monty.json import MSONable, jsanitize
from monty.os import cd
from scipy import constants
from scipy.optimize import fsolve
from scipy.spatial import distance
from pymatgen.core.lattice import Lattice
from pymatgen.core.units import Energy, Length
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine, Kpoint
from pymatgen.electronic_structure.core import Orbital
from pymatgen.electronic_structure.dos import CompleteDos, Dos, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
def write_energy(self, output_file) -> None:
    """Writes the energy to an output file.

        Args:
            output_file: Filename
        """
    with open(output_file, mode='w') as file:
        file.write('test\n')
        file.write(f'{len(self._bs.kpoints)}\n')
        if self.run_type == 'FERMI':
            sign = -1.0 if self.cond_band else 1.0
            for i, kpt in enumerate(self._bs.kpoints):
                eigs = []
                eigs.append(Energy(self._bs.bands[Spin(self.spin)][self.band_nb][i] - self._bs.efermi, 'eV').to('Ry'))
                a, b, c = kpt.frac_coords
                file.write(f'{a:12.8f} {b:12.8f} {c:12.8f}{len(eigs)}\n')
                for e in eigs:
                    file.write(f'{sign * float(e):18.8f}\n')
        else:
            for i, kpt in enumerate(self._bs.kpoints):
                eigs = []
                spin_lst = [self.spin] if self.run_type == 'DOS' else self._bs.bands
                for spin in spin_lst:
                    nb_bands = int(math.floor(self._bs.nb_bands * (1 - self.cb_cut)))
                    for j in range(nb_bands):
                        eigs.append(Energy(self._bs.bands[Spin(spin)][j][i] - self._bs.efermi, 'eV').to('Ry'))
                eigs.sort()
                if self.run_type == 'DOS' and self._bs.is_spin_polarized:
                    eigs.insert(0, self._ll)
                    eigs.append(self._hl)
                a, b, c = kpt.frac_coords
                file.write(f'{a:12.8f} {b:12.8f} {c:12.8f} {len(eigs)}\n')
                for e in eigs:
                    file.write(f'{float(e):18.8f}\n')