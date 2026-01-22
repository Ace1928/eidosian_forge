from __future__ import annotations
import warnings
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable, jsanitize
from pymatgen.core.structure import Molecule, Structure

        Returns pymatgen molecule from ASE Atoms.

        Args:
            atoms: ASE Atoms object
            cls: The Molecule class to instantiate (defaults to pymatgen molecule)
            **cls_kwargs: Any additional kwargs to pass to the cls

        Returns:
            Molecule: Equivalent pymatgen.core.structure.Molecule
        