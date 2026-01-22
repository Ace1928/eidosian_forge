from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
import matplotlib as mpl
from seaborn._marks.base import (
def _handle_capstyle(self, kws, vals):
    if vals['linestyle'][1] is None:
        capstyle = kws.get('solid_capstyle', mpl.rcParams['lines.solid_capstyle'])
        kws['dash_capstyle'] = capstyle