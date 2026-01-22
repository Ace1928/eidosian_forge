from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
@dataclass
class legend_justifications_view:
    """
    Global holder for how the legends should be justified
    """
    left: float = 0.5
    right: float = 0.5
    top: float = 0.5
    bottom: float = 0.5
    inside: Optional[TupleFloat2] = None