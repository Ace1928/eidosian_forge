from __future__ import annotations
import datetime
import itertools
import logging
import math
import os
import re
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from glob import glob
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.io import reverse_readfile, zopen
from monty.json import MSONable, jsanitize
from monty.os.path import zpath
from monty.re import regrep
from numpy.testing import assert_allclose
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.core.units import unitized
from pymatgen.electronic_structure.bandstructure import (
from pymatgen.electronic_structure.core import Magmom, Orbital, OrbitalType, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.common import VolumetricData as BaseVolumetricData
from pymatgen.io.core import ParseError
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.wannier90 import Unk
from pymatgen.util.io_utils import clean_lines, micro_pyawk
from pymatgen.util.num import make_symmetric_matrix_from_upper_tri
def write_unks(self, directory: str) -> None:
    """
        Write the UNK files to the given directory.

        Writes the cell-periodic part of the bloch wavefunctions from the
        WAVECAR file to each of the UNK files. There will be one UNK file for
        each of the kpoints in the WAVECAR file.

        Note:
            wannier90 expects the full kpoint grid instead of the symmetry-
            reduced one that VASP stores the wavefunctions on. You should run
            a nscf calculation with ISYM=0 to obtain the correct grid.

        Args:
            directory (str): directory where the UNK files are written
        """
    out_dir = Path(directory).expanduser()
    if not out_dir.exists():
        out_dir.mkdir(parents=False)
    elif not out_dir.is_dir():
        raise ValueError('invalid directory')
    N = np.prod(self.ng)
    for ik in range(self.nk):
        fname = f'UNK{ik + 1:05d}.'
        if self.vasp_type.lower()[0] == 'n':
            data = np.empty((self.nb, 2, *self.ng), dtype=np.complex128)
            for ib in range(self.nb):
                data[ib, 0, :, :, :] = np.fft.ifftn(self.fft_mesh(ik, ib, spinor=0)) * N
                data[ib, 1, :, :, :] = np.fft.ifftn(self.fft_mesh(ik, ib, spinor=1)) * N
            Unk(ik + 1, data).write_file(str(out_dir / (fname + 'NC')))
        else:
            data = np.empty((self.nb, *self.ng), dtype=np.complex128)
            for ispin in range(self.spin):
                for ib in range(self.nb):
                    data[ib, :, :, :] = np.fft.ifftn(self.fft_mesh(ik, ib, spin=ispin)) * N
                Unk(ik + 1, data).write_file(str(out_dir / f'{fname}{ispin + 1}'))