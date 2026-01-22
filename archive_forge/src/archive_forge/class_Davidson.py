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
class Davidson(Section):
    """Parameters for davidson diagonalization."""

    def __init__(self, new_prec_each: int=20, preconditioner: str='FULL_SINGLE_INVERSE', keywords: dict | None=None, subsections: dict | None=None, **kwargs):
        """
        Args:
            new_prec_each (int): How often to recalculate the preconditioner.
            preconditioner (str): Preconditioner to use.
                "FULL_ALL": Most effective state selective preconditioner based on diagonalization,
                    requires the ENERGY_GAP parameter to be an underestimate of the HOMO-LUMO gap.
                    This preconditioner is recommended for almost all systems, except very large
                    systems where make_preconditioner would dominate the total computational cost.
                "FULL_KINETIC": Cholesky inversion of S and T, fast construction, robust, use for
                    very large systems.
                "FULL_SINGLE": Based on H-eS diagonalization, not as good as FULL_ALL, but
                    somewhat cheaper to apply.
                "FULL_SINGLE_INVERSE": Based on H-eS cholesky inversion, similar to FULL_SINGLE
                    in preconditioning efficiency but cheaper to construct, might be somewhat
                    less robust. Recommended for large systems.
                "FULL_S_INVERSE": Cholesky inversion of S, not as good as FULL_KINETIC,
                    yet equally expensive.
                "NONE": skip preconditioning
            keywords: additional keywords
            subsections: additional subsections.
        """
        self.new_prec_each = new_prec_each
        self.preconditioner = preconditioner
        keywords = keywords or {}
        subsections = subsections or {}
        _keywords = {'NEW_PREC_EACH': Keyword('NEW_PREC_EACH', new_prec_each), 'PRECONDITIONER': Keyword('PRECONDITIONER', preconditioner)}
        keywords.update(_keywords)
        super().__init__('DAVIDSON', keywords=keywords, repeats=False, location=None, subsections=subsections, **kwargs)