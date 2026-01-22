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
class VaspPropertyParser(ABC):
    NAME = None

    @classmethod
    def get_name(cls):
        """Name of parser. Override the NAME constant in the class to specify a custom name,
        otherwise the class name is used"""
        return cls.NAME or cls.__name__

    @abstractmethod
    def has_property(self, cursor: _CURSOR, lines: _CHUNK) -> bool:
        """Function which checks if a property can be derived from a given
        cursor position"""

    @staticmethod
    def get_line(cursor: _CURSOR, lines: _CHUNK) -> str:
        """Helper function to get a line, and apply the check_line function"""
        return _check_line(lines[cursor])

    @abstractmethod
    def parse(self, cursor: _CURSOR, lines: _CHUNK) -> _RESULT:
        """Extract a property from the cursor position.
        Assumes that "has_property" would evaluate to True
        from cursor position """