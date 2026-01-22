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
class Magmoms(SimpleVaspChunkParser):
    """Get the x-component of the magnitization.
    This is just the magmoms in the collinear case.
    
    non-collinear spin is (currently) not supported"""
    LINE_DELIMITER = 'magnetization (x)'

    def parse(self, cursor: _CURSOR, lines: _CHUNK) -> _RESULT:
        natoms = self.get_from_header('natoms')
        nskip = 4
        magmoms = np.zeros(natoms)
        for i in range(natoms):
            line = self.get_line(cursor + i + nskip, lines)
            magmoms[i] = float(line.split()[-1])
        return {'magmoms': magmoms}