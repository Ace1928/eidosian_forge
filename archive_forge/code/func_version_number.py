from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def version_number(self) -> str:
    """The commit hash for the FHI-aims version."""
    line_start = self.reverse_search_for(['FHI-aims version'])
    if line_start == LINE_NOT_FOUND:
        raise AimsParseError('This file does not appear to be an aims-output file')
    return self.lines[line_start].split(':')[1].strip()