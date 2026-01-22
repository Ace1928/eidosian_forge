from __future__ import annotations
import itertools
import warnings
from collections.abc import Iterator, Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Composition, DummySpecies, Element, Lattice, Molecule, Species, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.outputs import Vasprun, Xdatcar
@staticmethod
def _combine_site_props(prop1: SitePropsType | None, prop2: SitePropsType | None, len1: int, len2: int) -> SitePropsType | None:
    """Combine site properties.

        Either one of prop1 or prop2 can be None, dict, or a list of dict. All
        possibilities of combining them are considered.
        """
    if prop1 is prop2 is None:
        return None
    if isinstance(prop1, dict) and prop1 == prop2:
        return prop1
    assert prop1 is None or isinstance(prop1, (list, dict))
    assert prop2 is None or isinstance(prop2, (list, dict))
    p1_candidates = {'NoneType': [None] * len1, 'dict': [prop1] * len1, 'list': prop1}
    p2_candidates = {'NoneType': [None] * len2, 'dict': [prop2] * len2, 'list': prop2}
    p1_selected: list = p1_candidates[type(prop1).__name__]
    p2_selected: list = p2_candidates[type(prop2).__name__]
    return p1_selected + p2_selected