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