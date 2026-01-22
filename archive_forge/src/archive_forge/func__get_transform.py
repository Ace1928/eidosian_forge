from __future__ import annotations
import re
from copy import copy
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Tuple, Optional, ClassVar
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import (
from matplotlib.dates import (
from matplotlib.axis import Axis
from matplotlib.scale import ScaleBase
from pandas import Series
from seaborn._core.rules import categorical_order
from seaborn._core.typing import Default, default
from typing import TYPE_CHECKING
def _get_transform(self):
    arg = self.trans

    def get_param(method, default):
        if arg == method:
            return default
        return float(arg[len(method):])
    if arg is None:
        return _make_identity_transforms()
    elif isinstance(arg, tuple):
        return arg
    elif isinstance(arg, str):
        if arg == 'ln':
            return _make_log_transforms()
        elif arg == 'logit':
            base = get_param('logit', 10)
            return _make_logit_transforms(base)
        elif arg.startswith('log'):
            base = get_param('log', 10)
            return _make_log_transforms(base)
        elif arg.startswith('symlog'):
            c = get_param('symlog', 1)
            return _make_symlog_transforms(c)
        elif arg.startswith('pow'):
            exp = get_param('pow', 2)
            return _make_power_transforms(exp)
        elif arg == 'sqrt':
            return _make_sqrt_transforms()
        else:
            raise ValueError(f'Unknown value provided for trans: {arg!r}')