from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants
import scipy.special
from monty.json import MSONable
from tqdm import tqdm
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Vasprun, Waveder
def _try_reading(dtypes):
    """Return None if failed."""
    for dtype in dtypes:
        try:
            return Waveder.from_binary(f'{directory}/WAVEDER', data_type=dtype)
        except ValueError as exc:
            if 'reshape' in str(exc):
                continue
            raise exc
    return None