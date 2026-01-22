from __future__ import annotations
import typing
from dataclasses import dataclass
import numpy as np
from ..hsluv import rgb_to_hex
from ._colormap import ColorMap, ColorMapKind

    Gradient colormap by calculating RGB colors independently

    The input data is the same as Matplotlib's LinearSegmentedColormap
    data whose values for each channel are functions.
    