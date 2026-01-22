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
def _write_input(self, input_dir: str='.') -> None:
    """
        Write the packmol input file to the input directory.

        Args:
            input_dir (str): path to the input directory
        """
    with open(f'{input_dir}/{self.input_file}', mode='w', encoding='utf-8') as inp:
        for key, val in self.control_params.items():
            inp.write(f'{key} {self._format_param_val(val)}\n')
        for idx, mol in enumerate(self.mols):
            filename = os.path.join(input_dir, f'{idx}.{self.control_params['filetype']}')
            if self.control_params['filetype'] == 'pdb':
                self.write_pdb(mol, filename, num=idx + 1)
            else:
                a = BabelMolAdaptor(mol)
                pm = pybel.Molecule(a.openbabel_mol)
                pm.write(self.control_params['filetype'], filename=filename, overwrite=True)
            inp.write('\n')
            inp.write(f'structure {os.path.join(input_dir, str(idx))}.{self.control_params['filetype']}\n')
            for key, val in self.param_list[idx].items():
                inp.write(f'  {key} {self._format_param_val(val)}\n')
            inp.write('end structure\n')