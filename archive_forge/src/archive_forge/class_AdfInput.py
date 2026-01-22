from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
from monty.io import reverse_readline
from monty.itertools import chunks
from monty.json import MSONable
from monty.serialization import zopen
from pymatgen.core.structure import Molecule
class AdfInput:
    """A basic ADF input file writer."""

    def __init__(self, task):
        """
        Initialization method.

        Args:
            task (AdfTask): An ADF task.
        """
        self.task = task

    def write_file(self, molecule, inp_file):
        """
        Write an ADF input file.

        Args:
            molecule (Molecule): The molecule for this task.
        inpfile (str): The name where the input file will be saved.
        """
        mol_blocks = []
        atom_block = AdfKey('Atoms', options=['cartesian'])
        for site in molecule:
            atom_block.add_subkey(AdfKey(str(site.specie), list(site.coords)))
        mol_blocks.append(atom_block)
        if molecule.charge != 0:
            net_q = molecule.charge
            ab = molecule.spin_multiplicity - 1
            charge_block = AdfKey('Charge', [net_q, ab])
            mol_blocks.append(charge_block)
            if ab != 0:
                unres_block = AdfKey('Unrestricted')
                mol_blocks.append(unres_block)
        with open(inp_file, 'w+') as file:
            for block in mol_blocks:
                file.write(str(block) + '\n')
            file.write(str(self.task) + '\n')
            file.write('END INPUT')