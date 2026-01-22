from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
class AimsParseError(Exception):
    """Exception raised if an error occurs when parsing an Aims output file."""

    def __init__(self, message: str) -> None:
        """Initialize the error with the message, message"""
        self.message = message
        super().__init__(self.message)