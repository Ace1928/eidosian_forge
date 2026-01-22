from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from string import Template
from monty.io import zopen
from pymatgen.core import Structure
from pymatgen.io.core import InputGenerator
from pymatgen.io.lammps.data import CombinedData, LammpsData
from pymatgen.io.lammps.inputs import LammpsInputFile
from pymatgen.io.lammps.sets import LammpsInputSet
@dataclass
class BaseLammpsGenerator(InputGenerator):
    """
    Base class to generate LAMMPS input sets.
    Uses template files for the input. The variables that can be changed
    in the input template file are those starting with a $ sign, e.g., $nsteps.
    This generic class is specialized for each template in subclasses, e.g. LammpsMinimization.
    You can create a template for your own task following those present in pymatgen/io/lammps/templates.
    The parameters are then replaced based on the values found
    in the settings dictionary that you provide, e.g., `{"nsteps": 1000}`.

    Attributes:
        template: Path (string) to the template file used to create the InputFile for LAMMPS.
        calc_type: Human-readable string used to briefly describe the type of computations performed by LAMMPS.
        settings: Dictionary containing the values of the parameters to replace in the template.
        keep_stages: If True, the string is formatted in a block structure with stage names
        and newlines that differentiate commands in the respective stages of the InputFile.
        If False, stage names are not printed and all commands appear in a single block.

    /!\\ This InputSet and InputGenerator implementation is based on templates and is not intended to be very flexible.
    For instance, pymatgen will not detect whether a given variable should be adapted based on others
    (e.g., the number of steps from the temperature), it will not check for convergence nor will it actually run LAMMPS.
    For additional flexibility and automation, use the atomate2-lammps implementation
    (https://github.com/Matgenix/atomate2-lammps).
    """
    inputfile: LammpsInputFile | None = field(default=None)
    template: str = field(default_factory=str)
    data: LammpsData | CombinedData | None = field(default=None)
    settings: dict = field(default_factory=dict)
    calc_type: str = field(default='lammps')
    keep_stages: bool = field(default=True)

    def get_input_set(self, structure: Structure | LammpsData | CombinedData) -> LammpsInputSet:
        """Generate a LammpsInputSet from the structure/data, tailored to the template file."""
        data: LammpsData = LammpsData.from_structure(structure) if isinstance(structure, Structure) else structure
        with zopen(self.template, mode='r') as file:
            template_str = file.read()
        input_str = Template(template_str).safe_substitute(**self.settings)
        input_file = LammpsInputFile.from_str(input_str, keep_stages=self.keep_stages)
        return LammpsInputSet(inputfile=input_file, data=data, calc_type=self.calc_type, template_file=self.template)