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
class AtomicMetadata(MSONable):
    """
    Metadata for basis sets and potentials in cp2k.

    Attributes:
        info: Info about this object
        element: Element for this object
        potential: The potential for this object
        name: Name of the object
        alias_names: Optional aliases
        filename: Name of the file containing this object
        version: Version
    """
    info: BasisInfo | PotentialInfo | None = None
    element: Element | None = None
    potential: Literal['All Electron', 'Pseudopotential'] | None = None
    name: str | None = None
    alias_names: list = field(default_factory=list)
    filename: str | None = None
    version: str | None = None

    def softmatch(self, other):
        """
        Soft matching to see if a desired basis/potential matches requirements.

        Does soft matching on the "info" attribute first. Then soft matches against the
        element and name/aliases.
        """
        if not isinstance(other, type(self)):
            return False
        if self.info and (not self.info.softmatch(other.info)):
            return False
        if self.element is not None and self.element != other.element:
            return False
        if self.potential is not None and self.potential != other.potential:
            return False
        this_names = [self.name]
        if self.alias_names:
            this_names.extend(self.alias_names)
        other_names = [other.name]
        if other.alias_names:
            other_names.extend(other.alias_names)
        return all((not (nm is not None and nm not in other_names) for nm in this_names))

    def get_hash(self) -> str:
        """Get a hash of this object."""
        md5 = hashlib.new('md5', usedforsecurity=False)
        md5.update(self.get_str().lower().encode('utf-8'))
        return md5.hexdigest()

    def get_str(self) -> str:
        """Get string representation."""
        return str(self)