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
def _check_line(line: str) -> str:
    """Auxiliary check line function for OUTCAR numeric formatting.
    See issue #179, https://gitlab.com/ase/ase/issues/179
    Only call in cases we need the numeric values
    """
    if re.search('[0-9]-[0-9]', line):
        line = re.sub('([0-9])-([0-9])', '\\1 -\\2', line)
    return line