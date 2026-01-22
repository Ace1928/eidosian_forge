from abc import ABC, abstractmethod
from typing import (Dict, Any, Sequence, TextIO, Iterator, Optional, Union,
import re
from warnings import warn
from pathlib import Path, PurePath
import numpy as np
import ase
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import ParseError, read
from ase.io.utils import ImageChunk
from ase.calculators.singlepoint import SinglePointDFTCalculator, SinglePointKPoint
class Stress(SimpleVaspChunkParser):
    """Process the stress from an OUTCAR"""
    LINE_DELIMITER = 'in kB '

    def parse(self, cursor: _CURSOR, lines: _CHUNK) -> _RESULT:
        line = self.get_line(cursor, lines)
        result = None
        try:
            stress = [float(a) for a in line.split()[2:]]
        except ValueError:
            warn('Found badly formatted stress line. Setting stress to None.')
        else:
            result = convert_vasp_outcar_stress(stress)
        return {'stress': result}