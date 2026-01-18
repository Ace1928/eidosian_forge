from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from functools import cached_property
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from warnings import warn
import numpy as np
import pandas as pd
from mizani.bounds import rescale
from .._utils import get_opposite_side
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import rename_aesthetics
from ..scales.scale_continuous import scale_continuous
from .guide import GuideElements, guide
@cached_property
def text_position(self) -> SidePosition:
    if not (position := self.theme.getp('legend_text_position')):
        position = 'right' if self.is_vertical else 'bottom'
    if self.is_vertical and position not in ('right', 'left'):
        msg = 'The text position for a vertical legend must be either left or right.'
        raise PlotnineError(msg)
    elif self.is_horizontal and position not in ('bottom', 'top'):
        msg = 'The text position for a horizonta legend must be either top or bottom.'
        raise PlotnineError(msg)
    return position