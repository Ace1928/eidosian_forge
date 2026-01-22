from __future__ import annotations
import typing
from dataclasses import dataclass
import numpy as np
from ..hsluv import rgb_to_hex
from ._colormap import ColorMap, ColorMapKind

        Return n colors from the gradient

        Parameters
        ----------
        n :
            Number of colors to return from the gradient.
        