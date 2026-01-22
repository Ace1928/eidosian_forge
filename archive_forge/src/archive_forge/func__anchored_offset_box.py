from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, fields
from functools import cached_property
from typing import TYPE_CHECKING, Literal, cast
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import ensure_xy_location
from .._utils.registry import Registry
from ..exceptions import PlotnineError, PlotnineWarning
from ..iapi import (
from ..mapping.aes import rename_aesthetics
from .guide import guide
def _anchored_offset_box(boxes: list[PackerBase]):
    """
            Put a group of guides into a single box for drawing
            """
    packer = lookup[elements.box]
    box = packer(children=boxes, align=elements.box_just, pad=elements.box_margin, sep=elements.spacing)
    return FlexibleAnchoredOffsetbox(xy_loc=(0.5, 0.5), child=box, pad=1, frameon=False, prop=FontProperties(size=0, stretch=0), bbox_to_anchor=(0, 0), bbox_transform=self.plot.figure.transFigure, borderpad=0.0, zorder=99.1)