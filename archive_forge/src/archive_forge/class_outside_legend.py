from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
@dataclass
class outside_legend:
    """
    What is required to layout an outside legend
    """
    box: FlexibleAnchoredOffsetbox
    justification: float