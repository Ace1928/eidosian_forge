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
class TypeParser(ABC):
    """Base class for parsing a type, e.g. header or chunk, 
    by applying the internal attached parsers"""

    def __init__(self, parsers):
        self.parsers = parsers

    @property
    def parsers(self):
        return self._parsers

    @parsers.setter
    def parsers(self, new_parsers) -> None:
        self._check_parsers(new_parsers)
        self._parsers = new_parsers

    @abstractmethod
    def _check_parsers(self, parsers) -> None:
        """Check the parsers are of correct type"""

    def parse(self, lines) -> _RESULT:
        """Execute the attached paresers, and return the parsed properties"""
        properties = {}
        for cursor, _ in enumerate(lines):
            for parser in self.parsers:
                if parser.has_property(cursor, lines):
                    prop = parser.parse(cursor, lines)
                    properties.update(prop)
        return properties