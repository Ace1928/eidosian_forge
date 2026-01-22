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
class LammpsMinimization(BaseLammpsGenerator):
    """
    Generator that yields a LammpsInputSet tailored for minimizing the energy of a system by iteratively
    adjusting atom coordinates.
    Example usage:
    ```
    structure = Structure.from_file("mp-149.cif")
    lmp_minimization = LammpsMinimization(units="atomic").get_input_set(structure)
    ```.

    Do not forget to specify the force field, otherwise LAMMPS will not be able to run!

    This InputSet and InputGenerator implementation is based on templates and is not intended to be very flexible.
    For instance, pymatgen will not detect whether a given variable should be adapted based on others
    (e.g., the number of steps from the temperature), it will not check for convergence nor will it actually run LAMMPS.
    For additional flexibility and automation, use the atomate2-lammps implementation
    (https://github.com/Matgenix/atomate2-lammps).
    """

    def __init__(self, template: str | None=None, units: str='metal', atom_style: str='full', dimension: int=3, boundary: str='p p p', read_data: str='system.data', force_field: str='Unspecified force field!', keep_stages: bool=False) -> None:
        """
        Args:
            template: Path (string) to the template file used to create the InputFile for LAMMPS.
            units: units to be used for the LAMMPS calculation (see LAMMPS docs).
            atom_style: atom_style to be used for the LAMMPS calculation (see LAMMPS docs).
            dimension: dimension to be used for the LAMMPS calculation (see LAMMPS docs).
            boundary: boundary to be used for the LAMMPS calculation (see LAMMPS docs).
            read_data: read_data to be used for the LAMMPS calculation (see LAMMPS docs).
            force_field: force field to be used for the LAMMPS calculation (see LAMMPS docs).
                Note that you should provide all the required information as a single string.
                In case of multiple lines expected in the input file,
                separate them with '
' in force_field.
            keep_stages: If True, the string is formatted in a block structure with stage names
                and newlines that differentiate commands in the respective stages of the InputFile.
                If False, stage names are not printed and all commands appear in a single block.
        """
        if template is None:
            template = f'{template_dir}/minimization.template'
        settings = {'units': units, 'atom_style': atom_style, 'dimension': dimension, 'boundary': boundary, 'read_data': read_data, 'force_field': force_field}
        super().__init__(template=template, settings=settings, calc_type='minimization', keep_stages=keep_stages)

    @property
    def units(self) -> str:
        """Return the argument of the command 'units' passed to the generator."""
        return self.settings['units']

    @property
    def atom_style(self) -> str:
        """Return the argument of the command 'atom_style' passed to the generator."""
        return self.settings['atom_style']

    @property
    def dimension(self) -> int:
        """Return the argument of the command 'dimension' passed to the generator."""
        return self.settings['dimension']

    @property
    def boundary(self) -> str:
        """Return the argument of the command 'boundary' passed to the generator."""
        return self.settings['boundary']

    @property
    def read_data(self) -> str:
        """Return the argument of the command 'read_data' passed to the generator."""
        return self.settings['read_data']

    @property
    def force_field(self) -> str:
        """Return the details of the force field commands passed to the generator."""
        return self.settings['force_field']