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
class ChunkParser(TypeParser, ABC):

    def __init__(self, parsers, header=None):
        super().__init__(parsers)
        self.header = header

    @property
    def header(self) -> _HEADER:
        return self._header

    @header.setter
    def header(self, value: Optional[_HEADER]) -> None:
        self._header = value or {}
        self.update_parser_headers()

    def update_parser_headers(self) -> None:
        """Apply the header to all available parsers"""
        for parser in self.parsers:
            parser.header = self.header

    def _check_parsers(self, parsers: Sequence[VaspChunkPropertyParser]) -> None:
        """Check the parsers are of correct type 'VaspChunkPropertyParser'"""
        if not all((isinstance(parser, VaspChunkPropertyParser) for parser in parsers)):
            raise TypeError('All parsers must be of type VaspChunkPropertyParser')

    @abstractmethod
    def build(self, lines: _CHUNK) -> Atoms:
        """Construct an atoms object of the chunk from the parsed results"""