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
class AimsInputSet(InputSet):
    """A class to represent a set of Aims inputs."""

    def __init__(self, parameters: dict[str, Any], structure: Structure | Molecule, properties: Sequence[str]=('energy', 'free_energy')) -> None:
        """Construct the AimsInputSet.

        Args:
            parameters (dict[str, Any]): The ASE parameters object for the calculation
            structure (Structure or Molecule): The Structure/Molecule objects to
                create the inputs for
            properties (Sequence[str]): The properties to calculate for the calculation
        """
        self._parameters = parameters
        self._structure = structure
        self._properties = properties
        aims_control_in, aims_geometry_in = self.get_input_files()
        super().__init__(inputs={CONTROL_FILE_NAME: aims_control_in, GEOMETRY_FILE_NAME: aims_geometry_in, PARAMS_JSON_FILE_NAME: json.dumps(self._parameters, cls=MontyEncoder)})

    def get_input_files(self) -> tuple[str, str]:
        """Get the input file contents for the calculation.

        Returns:
            tuple[str, str]: The contents of the control.in and geometry.in file
        """
        property_flags = {'forces': 'compute_forces', 'stress': 'compute_analytical_stress', 'stresses': 'compute_heat_flux'}
        updated_params = dict(**self._parameters)
        for prop in self._properties:
            aims_name = property_flags.get(prop)
            if aims_name is not None:
                updated_params[aims_name] = True
        aims_geometry_in = AimsGeometryIn.from_structure(self._structure)
        aims_control_in = AimsControlIn(updated_params)
        return (aims_control_in.get_content(structure=self._structure), aims_geometry_in.content)

    @property
    def control_in(self) -> str | slice | InputFile:
        """Get the control.in file contents."""
        return self[CONTROL_FILE_NAME]

    @property
    def geometry_in(self) -> str | slice | InputFile:
        """Get the geometry.in file contents."""
        return self[GEOMETRY_FILE_NAME]

    @property
    def params_json(self) -> str | slice | InputFile:
        """Get the JSON representation of the parameters dict."""
        return self[PARAMS_JSON_FILE_NAME]

    def set_parameters(self, *args, **kwargs) -> dict[str, Any]:
        """Set the parameters object for the AimsTemplate.

        This sets the parameters object that is passed to an AimsTemplate and
        resets the control.in file

        One can pass a dictionary mapping the aims variables to their values or
        the aims variables as keyword arguments. A combination of the two
        options is also allowed.

        Returns:
            dict[str, Any]: dictionary with the variables that have been added.
        """
        self._parameters.clear()
        for arg in args:
            self._parameters.update(arg)
        self._parameters.update(kwargs)
        aims_control_in, _ = self.get_input_files()
        self.inputs[CONTROL_FILE_NAME] = aims_control_in
        self.inputs[PARAMS_JSON_FILE_NAME] = json.dumps(self._parameters, cls=MontyEncoder)
        inputs = {str(key): val for key, val in self.inputs.items()}
        self.__dict__.update(inputs)
        return self._parameters

    def remove_parameters(self, keys: Iterable[str] | str, strict: bool=True) -> dict[str, Any]:
        """Remove the aims parameters listed in keys.

        This removes the aims variables from the parameters object.

        Args:
            keys (Iterable[str] or str): string or list of strings with the names of
                the aims parameters to be removed.
            strict (bool): whether to raise a KeyError if one of the aims parameters
                to be removed is not present.

        Returns:
            dict[str, Any]: Dictionary with the variables that have been removed.
        """
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            if key not in self._parameters:
                if strict:
                    raise ValueError(f'key={key!r} not in list(self._parameters)={list(self._parameters)!r}')
                continue
            del self._parameters[key]
        return self.set_parameters(**self._parameters)

    def set_structure(self, structure: Structure | Molecule):
        """Set the structure object for this input set.

        Args:
            structure (Structure or Molecule): The new Structure or Molecule
                for the calculation
        """
        self._structure = structure
        aims_control_in, aims_geometry_in = self.get_input_files()
        self.inputs[GEOMETRY_FILE_NAME] = aims_geometry_in
        self.inputs[CONTROL_FILE_NAME] = aims_control_in
        inputs = {str(key): val for key, val in self.inputs.items()}
        self.__dict__.update(inputs)