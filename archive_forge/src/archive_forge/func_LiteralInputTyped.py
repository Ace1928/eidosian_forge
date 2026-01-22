from __future__ import annotations
import asyncio
import inspect
import itertools
import json
import os
import sys
import textwrap
import types
from collections import defaultdict, namedtuple
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
from typing import (
import param
from param.parameterized import (
from param.reactive import rx
from .config import config
from .io import state
from .layout import (
from .pane import DataFrame as DataFramePane
from .pane.base import PaneBase, ReplacementPane
from .reactive import Reactive
from .util import (
from .util.checks import is_dataframe, is_mpl_axes, is_series
from .viewable import Layoutable, Viewable
from .widgets import (
from .widgets.button import _ButtonBase
def LiteralInputTyped(pobj: param.Parameter) -> Type[Widget]:
    if isinstance(pobj, param.Tuple):
        return type(str('TupleInput'), (LiteralInput,), {'type': tuple})
    elif isinstance(pobj, param.Number):
        return type(str('NumberInput'), (LiteralInput,), {'type': (int, float)})
    elif isinstance(pobj, param.Dict):
        return type(str('DictInput'), (LiteralInput,), {'type': dict})
    elif isinstance(pobj, param.List):
        return type(str('ListInput'), (LiteralInput,), {'type': list})
    return LiteralInput