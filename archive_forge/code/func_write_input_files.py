from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING
from monty.json import MSONable
def write_input_files(self):
    """Write input files to working directory."""
    self.molecule.to(filename=os.path.join(self.working_dir, self.coords_filename))
    if self.constraints:
        constrains_string = self.constrains_template(molecule=self.molecule, reference_fnm=self.coords_filename, constraints=self.constraints)
        with open('.constrains', mode='w') as file:
            file.write(constrains_string)