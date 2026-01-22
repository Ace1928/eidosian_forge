from __future__ import annotations
import os
import re
import shutil
import warnings
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import __version__ as CURRENT_VER
from pymatgen.io.core import InputFile
from pymatgen.io.lammps.data import CombinedData, LammpsData
from pymatgen.io.template import TemplateInputGen
def add_commands(self, stage_name: str, commands: str | list[str] | dict) -> None:
    """
        Method to add a LAMMPS commands and their arguments to a stage of
        the LammpsInputFile. The stage name should be provided: a default behavior
        is avoided here to avoid mistakes (e.g., the commands are added to the wrong stage).

        Example:
            In order to add the command ``pair_coeff 1 1 morse 0.0580 3.987 3.404``
            to the stage "Definition of the potential", simply use
            ```
            your_input_file.add_commands(
                stage_name="Definition of the potential",
                commands="pair_coeff 1 1 morse 0.0580 3.987 3.404"
            )
            ```
            To add multiple commands, use a dict or a list, e.g.,
            ```
            your_input_file.add_commands(
                stage_name="Definition of the potential",
                commands=["pair_coeff 1 1 morse 0.0580 3.987 3.404", "units atomic"]
            )
            your_input_file.add_commands(
                stage_name="Definition of the potential",
                commands={"pair_coeff": "1 1 morse 0.0580 3.987 3.404", "units": "atomic"}
            )
            ```

        Args:
            stage_name (str): name of the stage to which the command should be added.
            commands (str or list or dict): LAMMPS command, with or without the arguments.
        """
    if stage_name not in self.stages_names:
        raise ValueError('The provided stage name does not correspond to one of the LammpsInputFile.stages.')
    if isinstance(commands, str):
        self._add_command(command=commands, stage_name=stage_name)
    elif isinstance(commands, list):
        for comm in commands:
            if comm[0] == '#':
                self._add_comment(comment=comm[1:].strip(), inline=True, stage_name=stage_name, index_comment=True)
            else:
                self._add_command(command=comm, stage_name=stage_name)
    elif isinstance(commands, dict):
        for comm, args in commands.items():
            if comm[0] == '#':
                self._add_comment(comment=comm[1:].strip(), inline=True, stage_name=stage_name, index_comment=True)
            else:
                self._add_command(command=comm, args=args, stage_name=stage_name)
    else:
        raise TypeError('The command should be a string, list of strings or dictionary.')