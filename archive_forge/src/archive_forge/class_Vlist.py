from __future__ import annotations
import abc
import copy
import enum
import functools
import logging
import os
import re
import types
import unicodedata
import string
import typing as T
from typing import NamedTuple
import numpy as np
from pyparsing import (
import matplotlib as mpl
from . import cbook
from ._mathtext_data import (
from .font_manager import FontProperties, findfont, get_font
from .ft2font import FT2Font, FT2Image, KERNING_DEFAULT
from packaging.version import parse as parse_version
from pyparsing import __version__ as pyparsing_version
class Vlist(List):
    """A vertical list of boxes."""

    def __init__(self, elements: T.Sequence[Node], h: float=0.0, m: T.Literal['additional', 'exactly']='additional'):
        super().__init__(elements)
        self.vpack(h=h, m=m)

    def vpack(self, h: float=0.0, m: T.Literal['additional', 'exactly']='additional', l: float=np.inf) -> None:
        """
        Compute the dimensions of the resulting boxes, and to adjust the glue
        if one of those dimensions is pre-specified.

        Parameters
        ----------
        h : float, default: 0
            A height.
        m : {'exactly', 'additional'}, default: 'additional'
            Whether to produce a box whose height is 'exactly' *h*; or a box
            with the natural height of the contents, plus *h* ('additional').
        l : float, default: np.inf
            The maximum height.

        Notes
        -----
        The defaults produce a box with the natural height of the contents.
        """
        w = 0.0
        d = 0.0
        x = 0.0
        total_stretch = [0.0] * 4
        total_shrink = [0.0] * 4
        for p in self.children:
            if isinstance(p, Box):
                x += d + p.height
                d = p.depth
                if not np.isinf(p.width):
                    s = getattr(p, 'shift_amount', 0.0)
                    w = max(w, p.width + s)
            elif isinstance(p, Glue):
                x += d
                d = 0.0
                glue_spec = p.glue_spec
                x += glue_spec.width
                total_stretch[glue_spec.stretch_order] += glue_spec.stretch
                total_shrink[glue_spec.shrink_order] += glue_spec.shrink
            elif isinstance(p, Kern):
                x += d + p.width
                d = 0.0
            elif isinstance(p, Char):
                raise RuntimeError('Internal mathtext error: Char node found in Vlist')
        self.width = w
        if d > l:
            x += d - l
            self.depth = l
        else:
            self.depth = d
        if m == 'additional':
            h += x
        self.height = h
        x = h - x
        if x == 0:
            self.glue_sign = 0
            self.glue_order = 0
            self.glue_ratio = 0.0
            return
        if x > 0.0:
            self._set_glue(x, 1, total_stretch, 'Overful')
        else:
            self._set_glue(x, -1, total_shrink, 'Underful')