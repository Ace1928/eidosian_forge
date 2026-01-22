from __future__ import annotations
import gzip
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from pymatgen.core import Lattice, Molecule, Structure
@dataclass
class AimsControlIn(MSONable):
    """Class representing and FHI-aims control.in file

    Attributes:
        _parameters (dict[str, Any]): The parameters dictionary containing all input
            flags (key) and values for the control.in file
    """
    _parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize the output list of _parameters"""
        self._parameters.setdefault('output', [])

    def __getitem__(self, key: str) -> Any:
        """Get an input parameter

        Args:
            key (str): The parameter to get

        Returns:
            The setting for that parameter

        Raises:
            KeyError: If the key is not in self._parameters
        """
        if key not in self._parameters:
            raise KeyError(f'{key} not set in AimsControlIn')
        return self._parameters[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set an attribute of the class

        Args:
            key (str): The parameter to get
            value (Any): The value for that parameter
        """
        if key == 'output':
            if isinstance(value, str):
                value = [value]
            self._parameters[key] += value
        else:
            self._parameters[key] = value

    def __delitem__(self, key: str) -> Any:
        """Delete a parameter from the input object

        Args:
        key (str): The key in the parameter to remove

        Returns:
            Either the value of the deleted parameter or None if key is
            not in self._parameters
        """
        return self._parameters.pop(key, None)

    @property
    def parameters(self) -> dict[str, Any]:
        """The dictionary of input parameters for control.in"""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: dict[str, Any]) -> None:
        """Reset a control.in inputs from a parameters dictionary

        Args:
            parameters (dict[str, Any]): The new set of parameters to use
        """
        self._parameters = parameters
        self._parameters.setdefault('output', [])

    def get_aims_control_parameter_str(self, key: str, value: Any, fmt: str) -> str:
        """Get the string needed to add a parameter to the control.in file

        Args:
            key (str): The name of the input flag
            value (Any): The value to be set for the flag
            fmt (str): The format string to apply to the value

        Returns:
            str: The line to add to the control.in file
        """
        if value is None:
            return ''
        return f'{key:35s}{fmt % value}\n'

    def get_content(self, structure: Structure | Molecule, verbose_header: bool=False, directory: str | Path | None=None) -> str:
        """Get the content of the file

        Args:
            structure (Structure | Molecule): The structure to write the input
                file for
            verbose_header (bool): If True print the input option dictionary
            directory: str | Path | None = The directory for the calculation,

        Returns:
            str: The content of the file for a given structure
        """
        parameters = deepcopy(self._parameters)
        if directory is None:
            directory = ''
        lim = '#' + '=' * 79
        content = ''
        if parameters['xc'] == 'LDA':
            parameters['xc'] = 'pw-lda'
        cubes = parameters.pop('cubes', None)
        if verbose_header:
            content += '# \n# List of parameters used to initialize the calculator:'
            for param, val in parameters.items():
                content += f'#     {param}:{val}\n'
        content += lim + '\n'
        assert ('smearing' in parameters and 'occupation_type' in parameters) is False
        for key, value in parameters.items():
            if key in ['species_dir', 'plus_u']:
                continue
            if key == 'smearing':
                name = parameters['smearing'][0].lower()
                if name == 'fermi-dirac':
                    name = 'fermi'
                width = parameters['smearing'][1]
                if name == 'methfessel-paxton':
                    order = parameters['smearing'][2]
                    order = ' %d' % order
                else:
                    order = ''
                content += self.get_aims_control_parameter_str('occupation_type', (name, width, order), '%s %f%s')
            elif key == 'output':
                for output_type in value:
                    content += self.get_aims_control_parameter_str(key, output_type, '%s')
            elif key == 'vdw_correction_hirshfeld' and value:
                content += self.get_aims_control_parameter_str(key, '', '%s')
            elif isinstance(value, bool):
                content += self.get_aims_control_parameter_str(key, str(value).lower(), '.%s.')
            elif isinstance(value, (tuple, list)):
                content += self.get_aims_control_parameter_str(key, ' '.join(map(str, value)), '%s')
            elif isinstance(value, str):
                content += self.get_aims_control_parameter_str(key, value, '%s')
            else:
                content += self.get_aims_control_parameter_str(key, value, '%r')
        if cubes:
            for cube in cubes:
                content += cube.control_block
        content += lim + '\n\n'
        species_dir = self._parameters.get('species_dir', os.environ.get('AIMS_SPECIES_DIR'))
        content += self.get_species_block(structure, species_dir)
        return content

    def write_file(self, structure: Structure | Molecule, directory: str | Path | None=None, verbose_header: bool=False, overwrite: bool=False) -> None:
        """Writes the control.in file

        Args:
            structure (Structure | Molecule): The structure to write the input
                file for
            directory (str or Path): The directory to write the control.in file.
                If None use cwd
            verbose_header (bool): If True print the input option dictionary
            overwrite (bool): If True allow to overwrite existing files

        Raises:
            ValueError: If a file must be overwritten and overwrite is False
            ValueError: If k-grid is not provided for the periodic structures
        """
        directory = directory or Path.cwd()
        if (Path(directory) / 'control.in').exists() and (not overwrite):
            raise ValueError(f'control.in file already in {directory}')
        if isinstance(structure, Structure) and ('k_grid' not in self._parameters and 'k_grid_density' not in self._parameters):
            raise ValueError('k-grid must be defined for periodic systems')
        content = self.get_content(structure, verbose_header)
        with open(f'{directory}/control.in', mode='w') as file:
            file.write(f'#{'=' * 72}\n')
            file.write(f'# FHI-aims geometry file: {directory}/geometry.in\n')
            file.write('# File generated from pymatgen\n')
            file.write(f'# {time.asctime()}\n')
            file.write(f'#{'=' * 72}\n')
            file.write(content)

    def get_species_block(self, structure: Structure | Molecule, species_dir: str | Path) -> str:
        """Get the basis set information for a structure

        Args:
            structure (Molecule or Structure): The structure to get the basis set information for
            species_dir (str or Pat:): The directory to find the species files in

        Returns:
            The block to add to the control.in file for the species

        Raises:
            ValueError: If a file for the species is not found
        """
        block = ''
        species = np.unique(structure.species)
        for sp in species:
            filename = f'{species_dir}/{sp.Z:02d}_{sp.symbol}_default'
            if Path(filename).exists():
                with open(filename) as sf:
                    block += ''.join(sf.readlines())
            elif Path(f'{filename}.gz').exists():
                with gzip.open(f'{filename}.gz', mode='rt') as sf:
                    block += ''.join(sf.readlines())
            else:
                raise ValueError(f'Species file for {sp.symbol} not found.')
        return block

    def as_dict(self) -> dict[str, Any]:
        """Get a dictionary representation of the geometry.in file."""
        dct: dict[str, Any] = {}
        dct['@module'] = type(self).__module__
        dct['@class'] = type(self).__name__
        dct['parameters'] = self.parameters
        return dct

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> Self:
        """Initialize from dictionary.

        Args:
            dct (dict[str, Any]): The MontyEncoded dictionary

        Returns:
            The AimsControlIn for dct
        """
        decoded = {key: MontyDecoder().process_decoded(val) for key, val in dct.items() if not key.startswith('@')}
        return cls(_parameters=decoded['parameters'])