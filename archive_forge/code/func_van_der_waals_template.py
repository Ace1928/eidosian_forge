from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def van_der_waals_template(radii: dict[str, float], mode: str='atomic') -> str:
    """
        Args:
            radii (dict): Dictionary with custom van der Waals radii, in
                Angstroms, keyed by either atomic number or sequential
                atom number (see 'mode' kwarg).
                Ex: {1: 1.20, 12: 1.70}
            mode: 'atomic' or 'sequential'. In 'atomic' mode (default), dict keys
                represent the atomic number associated with each radius (e.g., '12' = carbon).
                In 'sequential' mode, dict keys represent the sequential position of
                a single specific atom in the input structure.
                **NOTE: keys must be given as strings even though they are numbers!**.

        Returns:
            String representing Q-Chem input format for van_der_waals section
        """
    vdw_list = []
    vdw_list.append('$van_der_waals')
    if mode == 'atomic':
        vdw_list.append('1')
    elif mode == 'sequential':
        vdw_list.append('2')
    else:
        raise ValueError(f"Invalid mode={mode!r}, must be 'atomic' or 'sequential'")
    for num, radius in radii.items():
        vdw_list.append(f'   {num} {radius}')
    vdw_list.append('$end')
    return '\n'.join(vdw_list)