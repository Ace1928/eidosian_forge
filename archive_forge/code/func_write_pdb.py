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
@staticmethod
def write_pdb(mol: Molecule, filename: str, name: str | None=None, num=None) -> None:
    """Dump the molecule into pdb file with custom residue name and number."""
    with ScratchDir('.'):
        mol.to(fmt='pdb', filename='tmp.pdb')
        bma = BabelMolAdaptor.from_file('tmp.pdb', 'pdb')
    num = num or 1
    name = name or f'ml{num}'
    pbm = pybel.Molecule(bma._ob_mol)
    for x in pbm.residues:
        x.OBResidue.SetName(name)
        x.OBResidue.SetNum(num)
    pbm.write(format='pdb', filename=filename, overwrite=True)