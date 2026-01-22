from __future__ import annotations
import collections
import itertools
from typing import TYPE_CHECKING

    Warren-Crowley parameters.

    Args:
        structure: Pymatgen Structure.
        r: Radius
        dr: Shell width

    Returns:
        Warren-Crowley parameters in the form of a dict, e.g., {(Element Mo, Element W): -1.0, ...}
    