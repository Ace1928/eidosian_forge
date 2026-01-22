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
def _check_parsers(self, parsers: Sequence[VaspHeaderPropertyParser]) -> None:
    """Check the parsers are of correct type 'VaspHeaderPropertyParser'"""
    if not all((isinstance(parser, VaspHeaderPropertyParser) for parser in parsers)):
        raise TypeError('All parsers must be of type VaspHeaderPropertyParser')