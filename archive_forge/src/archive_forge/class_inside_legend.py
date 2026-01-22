from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
@dataclass
class inside_legend:
    """
    What is required to layout an inside legend
    """
    box: FlexibleAnchoredOffsetbox
    justification: TupleFloat2
    position: TupleFloat2