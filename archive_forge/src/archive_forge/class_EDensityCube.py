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
class EDensityCube(Section):
    """Controls printing of the electron density cube file."""

    def __init__(self, keywords: dict | None=None, subsections: dict | None=None, **kwargs):
        keywords = keywords or {}
        subsections = subsections or {}
        description = 'Controls the printing of cube files with the electronic density and, for LSD calculations, the spin density.'
        super().__init__('E_DENSITY_CUBE', subsections=subsections, description=description, keywords=keywords, **kwargs)