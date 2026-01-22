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
class VaspChunkPropertyParser(VaspPropertyParser, ABC):
    """Base class for parsing a chunk of the OUTCAR.
    The base assumption is that only a chunk of lines is passed"""

    def __init__(self, header: _HEADER=None):
        super().__init__()
        header = header or {}
        self.header = header

    def get_from_header(self, key: str) -> Any:
        """Get a key from the header, and raise a ParseError
        if that key doesn't exist"""
        try:
            return self.header[key]
        except KeyError:
            raise ParseError('Parser requested unavailable key "{}" from header'.format(key))