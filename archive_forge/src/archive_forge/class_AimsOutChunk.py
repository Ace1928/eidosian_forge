from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@dataclass
class AimsOutChunk:
    """Base class for AimsOutChunks.

    Attributes:
        lines (list[str]): The list of all lines in the chunk
    """
    lines: list[str] = field(default_factory=list)

    def reverse_search_for(self, keys: list[str], line_start: int=0) -> int:
        """Find the last time one of the keys appears in self.lines.

        Args:
            keys (list[str]): The key strings to search for in self.lines
            line_start (int): The lowest index to search for in self.lines

        Returns:
            The last time one of the keys appears in self.lines
        """
        for idx, line in enumerate(self.lines[line_start:][::-1]):
            if any((key in line for key in keys)):
                return len(self.lines) - idx - 1
        return LINE_NOT_FOUND

    def search_for_all(self, key: str, line_start: int=0, line_end: int=-1) -> list[int]:
        """Find the all times the key appears in self.lines.

        Args:
            key (str): The key string to search for in self.lines
            line_start (int): The first line to start the search from
            line_end (int): The last line to end the search at

        Returns:
            All times the key appears in the lines
        """
        line_index = []
        for ll, line in enumerate(self.lines[line_start:line_end]):
            if key in line:
                line_index.append(ll + line_start)
        return line_index

    def parse_scalar(self, property: str) -> float | None:
        """Parse a scalar property from the chunk.

        Args:
            property (str): The property key to parse

        Returns:
            The scalar value of the property or None if not found
        """
        line_start = self.reverse_search_for(SCALAR_PROPERTY_TO_LINE_KEY[property])
        if line_start == LINE_NOT_FOUND:
            return None
        line = self.lines[line_start]
        return float(line.split(':')[-1].strip().split()[0])