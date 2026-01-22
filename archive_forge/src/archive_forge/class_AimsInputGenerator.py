from __future__ import annotations
import copy
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from warnings import warn
import numpy as np
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.core import Molecule, Structure
from pymatgen.io.aims.inputs import AimsControlIn, AimsGeometryIn
from pymatgen.io.aims.parsers import AimsParseError, read_aims_output
from pymatgen.io.core import InputFile, InputGenerator, InputSet
@dataclass
class AimsInputGenerator(InputGenerator):
    """
    A class to generate Aims input sets.

    Parameters
    ----------
    user_params: dict[str, Any]
        Updates the default parameters for the FHI-aims calculator
    user_kpoints_settings: dict[str, Any]
        The settings used to create the k-grid parameters for FHI-aims
    """
    user_params: dict[str, Any] = field(default_factory=dict)
    user_kpoints_settings: dict[str, Any] = field(default_factory=dict)

    def get_input_set(self, structure: Structure | Molecule | None=None, prev_dir: str | Path | None=None, properties: list[str] | None=None) -> AimsInputSet:
        """Generate an AimsInputSet object.

        Parameters
        ----------
        structure : Structure or Molecule
            Structure or Molecule to generate the input set for.
        prev_dir: str or Path
            Path to the previous working directory
        properties: list[str]
            System properties that are being calculated

        Returns:
            AimsInputSet: The input set for the calculation of structure
        """
        prev_structure, prev_parameters, _ = self._read_previous(prev_dir)
        structure = structure or prev_structure
        if structure is None:
            raise ValueError('No structure can be determined to generate the input set')
        parameters = self._get_input_parameters(structure, prev_parameters)
        properties = self._get_properties(properties, parameters)
        return AimsInputSet(parameters=parameters, structure=structure, properties=properties)

    @staticmethod
    def _read_previous(prev_dir: str | Path | None=None) -> tuple[Structure | Molecule | None, dict[str, Any], dict[str, Any]]:
        """Read in previous results.

        Parameters
        ----------
        prev_dir: str or Path
            The previous directory for the calculation
        """
        prev_structure: Structure | Molecule | None = None
        prev_parameters = {}
        prev_results: dict[str, Any] = {}
        if prev_dir:
            split_prev_dir = str(prev_dir).split(':')[-1]
            with open(f'{split_prev_dir}/parameters.json') as param_file:
                prev_parameters = json.load(param_file, cls=MontyDecoder)
            try:
                aims_output: Sequence[Structure | Molecule] = read_aims_output(f'{split_prev_dir}/aims.out', index=slice(-1, None))
                prev_structure = aims_output[0]
                prev_results = prev_structure.properties
                prev_results.update(prev_structure.site_properties)
            except (IndexError, AimsParseError):
                pass
        return (prev_structure, prev_parameters, prev_results)

    @staticmethod
    def _get_properties(properties: list[str] | None=None, parameters: dict[str, Any] | None=None) -> list[str]:
        """Get the properties to calculate.

        Parameters
        ----------
        properties: list[str]
            The currently requested properties
        parameters: dict[str, Any]
            The parameters for this calculation

        Returns:
            list[str]: The list of properties to calculate
        """
        if properties is None:
            properties = ['energy', 'free_energy']
        if parameters is None:
            return properties
        if 'compute_forces' in parameters and 'forces' not in properties:
            properties.append('forces')
        if 'compute_heat_flux' in parameters and 'stresses' not in properties:
            properties.append('stress')
            properties.append('stresses')
        if 'stress' not in properties and ('compute_analytical_stress' in parameters or 'compute_numerical_stress' in parameters or 'compute_heat_flux' in parameters):
            properties.append('stress')
        return properties

    def _get_input_parameters(self, structure: Structure | Molecule, prev_parameters: dict[str, Any] | None=None) -> dict[str, Any]:
        """Create the input parameters.

        Parameters
        ----------
        structure: Structure | Molecule
            The structure or molecule for the system
        prev_parameters: dict[str, Any]
            The previous calculation's calculation parameters

        Returns:
            dict: The input object
        """
        parameters: dict[str, Any] = {'xc': 'pbe', 'relativistic': 'atomic_zora scalar'}
        prev_parameters = {} if prev_parameters is None else copy.deepcopy(prev_parameters)
        prev_parameters.pop('relax_geometry', None)
        prev_parameters.pop('relax_unit_cell', None)
        kpt_settings = copy.deepcopy(self.user_kpoints_settings)
        if isinstance(structure, Structure) and 'k_grid' in prev_parameters:
            density = self.k2d(structure, prev_parameters.pop('k_grid'))
            if 'density' not in kpt_settings:
                kpt_settings['density'] = density
        parameter_updates = self.get_parameter_updates(structure, prev_parameters)
        parameters = recursive_update(parameters, parameter_updates)
        parameters = recursive_update(parameters, self.user_params)
        if 'k_grid' in parameters and 'density' in kpt_settings:
            warn('WARNING: the k_grid is set in user_params and in the kpt_settings, using the one passed in user_params.', stacklevel=1)
        elif isinstance(structure, Structure) and 'k_grid' not in parameters:
            density = kpt_settings.get('density', 5.0)
            even = kpt_settings.get('even', True)
            parameters['k_grid'] = self.d2k(structure, density, even)
        elif isinstance(structure, Molecule) and 'k_grid' in parameters:
            warn('WARNING: removing unnecessary k_grid information', stacklevel=1)
            del parameters['k_grid']
        return parameters

    def get_parameter_updates(self, structure: Structure | Molecule, prev_parameters: dict[str, Any]) -> dict[str, Any]:
        """Update the parameters for a given calculation type.

        Parameters
        ----------
        structure : Structure or Molecule
            The system to run
        prev_parameters: dict[str, Any]
            Previous calculation parameters.

        Returns:
            dict: A dictionary of updates to apply.
        """
        return prev_parameters

    def d2k(self, structure: Structure, kptdensity: float | list[float]=5.0, even: bool=True) -> Iterable[float]:
        """Convert k-point density to Monkhorst-Pack grid size.

        inspired by [ase.calculators.calculator.kptdensity2monkhorstpack]

        Parameters
        ----------
        structure: Structure
            Contains unit cell and information about boundary conditions.
        kptdensity: float or list of floats
            Required k-point density.  Default value is 5.0 point per Ang^-1.
        even: bool
            Round up to even numbers.

        Returns:
            dict: Monkhorst-Pack grid size in all directions
        """
        recipcell = structure.lattice.inv_matrix
        return self.d2k_recipcell(recipcell, structure.lattice.pbc, kptdensity, even)

    def k2d(self, structure: Structure, k_grid: np.ndarray[int]):
        """Generate the kpoint density in each direction from given k_grid.

        Parameters
        ----------
        structure: Structure
            Contains unit cell and information about boundary conditions.
        k_grid: np.ndarray[int]
            k_grid that was used.

        Returns:
            dict: Density of kpoints in each direction. result.mean() computes average density
        """
        recipcell = structure.lattice.inv_matrix
        densities = k_grid / (2 * np.pi * np.sqrt((recipcell ** 2).sum(axis=1)))
        return np.array(densities)

    @staticmethod
    def d2k_recipcell(recipcell: np.ndarray, pbc: Sequence[bool], kptdensity: float | Sequence[float]=5.0, even: bool=True) -> Sequence[int]:
        """Convert k-point density to Monkhorst-Pack grid size.

        Parameters
        ----------
        recipcell: Cell
            The reciprocal cell
        pbc: Sequence[bool]
            If element of pbc is True then system is periodic in that direction
        kptdensity: float or list[floats]
            Required k-point density.  Default value is 3.5 point per Ang^-1.
        even: bool
            Round up to even numbers.

        Returns:
            dict: Monkhorst-Pack grid size in all directions
        """
        if not isinstance(kptdensity, Iterable):
            kptdensity = 3 * [float(kptdensity)]
        kpts: list[int] = []
        for i in range(3):
            if pbc[i]:
                k = 2 * np.pi * np.sqrt((recipcell[i] ** 2).sum()) * float(kptdensity[i])
                if even:
                    kpts.append(2 * int(np.ceil(k / 2)))
                else:
                    kpts.append(int(np.ceil(k)))
            else:
                kpts.append(1)
        return kpts