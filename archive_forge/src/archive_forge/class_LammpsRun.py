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
class LammpsRun(MSONable):
    """
    Examples for various simple LAMMPS runs with given simulation box,
    force field and a few more settings. Experienced LAMMPS users should
    consider using write_lammps_inputs method with more sophisticated
    templates.
    """

    def __init__(self, script_template: str, settings: dict, data: LammpsData | str, script_filename: str) -> None:
        """
        Base constructor.

        Args:
            script_template (str): String template for input script
                with placeholders. The format for placeholders has to
                be '$variable_name', e.g., '$temperature'
            settings (dict): Contains values to be written to the
                placeholders, e.g., {'temperature': 1}.
            data (LammpsData or str): Data file as a LammpsData
                instance or path to an existing data file. Default to
                None, i.e., no data file supplied. Useful only when
                read_data cmd is in the script.
            script_filename (str): Filename for the input script.
        """
        self.script_template = script_template
        self.settings = settings
        self.data = data
        self.script_filename = script_filename

    def write_inputs(self, output_dir: str, **kwargs) -> None:
        """
        Writes all input files (input script, and data if needed).
        Other supporting files are not handled at this moment.

        Args:
            output_dir (str): Directory to output the input files.
            **kwargs: kwargs supported by LammpsData.write_file.
        """
        write_lammps_inputs(output_dir=output_dir, script_template=self.script_template, settings=self.settings, data=self.data, script_filename=self.script_filename, **kwargs)

    @classmethod
    def md(cls, data: LammpsData | str, force_field: str, temperature: float, nsteps: int, other_settings: dict | None=None) -> LammpsRun:
        """
        Example for a simple MD run based on template md.template.

        Args:
            data (LammpsData or str): Data file as a LammpsData
                instance or path to an existing data file.
            force_field (str): Combined force field related cmds. For
                example, 'pair_style eam\\npair_coeff * * Cu_u3.eam'.
            temperature (float): Simulation temperature.
            nsteps (int): No. of steps to run.
            other_settings (dict): other settings to be filled into
                placeholders.
        """
        template_path = os.path.join(template_dir, 'md.template')
        with open(template_path, encoding='utf-8') as file:
            script_template = file.read()
        settings = other_settings.copy() if other_settings else {}
        settings.update({'force_field': force_field, 'temperature': temperature, 'nsteps': nsteps})
        script_filename = 'in.md'
        return cls(script_template=script_template, settings=settings, data=data, script_filename=script_filename)