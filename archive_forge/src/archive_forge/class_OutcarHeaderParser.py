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
class OutcarHeaderParser(HeaderParser):
    """Class for parsing a chunk of an OUTCAR."""

    def __init__(self, parsers: Sequence[VaspHeaderPropertyParser]=None, workdir: Union[str, PurePath]=None):
        global default_header_parsers
        parsers = parsers or default_header_parsers.make_parsers()
        super().__init__(parsers)
        self.workdir = workdir

    @property
    def workdir(self):
        return self._workdir

    @workdir.setter
    def workdir(self, value):
        if value is not None:
            value = Path(value)
        self._workdir = value

    def _build_symbols(self, results: _RESULT) -> Sequence[str]:
        if 'symbols' in results:
            return results.pop('symbols')
        for required_key in ('ion_types', 'species'):
            if required_key not in results:
                raise ParseError('Did not find required key "{}" in parsed header results.'.format(required_key))
        ion_types = results.pop('ion_types')
        species = results.pop('species')
        if len(ion_types) != len(species):
            raise ParseError('Expected length of ion_types to be same as species, but got ion_types={} and species={}'.format(len(ion_types), len(species)))
        symbols = []
        for n, sym in zip(ion_types, species):
            symbols.extend(n * [sym])
        return symbols

    def _get_constraint(self):
        """Try and get the constraints from the POSCAR of CONTCAR
        since they aren't located in the OUTCAR, and thus we cannot construct an
        OUTCAR parser which does this.
        """
        constraint = None
        if self.workdir is not None:
            constraint = read_constraints_from_file(self.workdir)
        return constraint

    def build(self, lines: _CHUNK) -> _RESULT:
        """Apply the header parsers, and build the header"""
        results = self.parse(lines)
        symbols = self._build_symbols(results)
        natoms = len(symbols)
        constraint = self._get_constraint()
        header = dict(symbols=symbols, natoms=natoms, constraint=constraint, **results)
        return header