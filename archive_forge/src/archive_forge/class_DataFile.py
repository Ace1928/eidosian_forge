from __future__ import annotations
import abc
import copy
import hashlib
import itertools
import os
import re
import textwrap
import typing
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element
from pymatgen.io.cp2k.utils import chunk, postprocessor, preprocessor
from pymatgen.io.vasp.inputs import Kpoints as VaspKpoints
from pymatgen.io.vasp.inputs import KpointsSupportedModes
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@dataclass
class DataFile(MSONable):
    """A data file for a cp2k calc."""
    objects: Sequence | None = None

    @classmethod
    def from_file(cls, filename) -> Self:
        """Load from a file, reserved for child classes."""
        with open(filename, encoding='utf-8') as file:
            data = cls.from_str(file.read())
            for obj in data.objects:
                obj.filename = filename
            return data

    @classmethod
    @abc.abstractmethod
    def from_str(cls, string: str) -> None:
        """Initialize from a string."""
        raise NotImplementedError

    def write_file(self, filename):
        """Write to a file."""
        with open(filename, mode='w', encoding='utf-8') as file:
            file.write(self.get_str())

    def get_str(self) -> str:
        """Get string representation."""
        return '\n'.join((b.get_str() for b in self.objects or []))

    def __str__(self):
        return self.get_str()