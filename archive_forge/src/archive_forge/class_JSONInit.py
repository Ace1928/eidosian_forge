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
class JSONInit(param.Parameterized):
    """
    Callable that can be passed to Widgets.initializer to set Parameter
    values using JSON. There are three approaches that may be used:
    1. If the json_file argument is specified, this takes precedence.
    2. The JSON file path can be specified via an environment variable.
    3. The JSON can be read directly from an environment variable.
    Here is an easy example of setting such an environment variable on
    the commandline:
    PARAM_JSON_INIT='{"p1":5}' jupyter notebook
    This addresses any JSONInit instances that are inspecting the
    default environment variable called PARAM_JSON_INIT, instructing it to set
    the 'p1' parameter to 5.
    """
    varname = param.String(default='PARAM_JSON_INIT', doc='\n        The name of the environment variable containing the JSON\n        specification.')
    target = param.String(default=None, doc='\n        Optional key in the JSON specification dictionary containing the\n        desired parameter values.')
    json_file = param.String(default=None, doc='\n        Optional path to a JSON file containing the parameter settings.')

    def __call__(self, parameterized):
        warnobj = param.main.param if isinstance(parameterized, type) else parameterized.param
        param_class = parameterized if isinstance(parameterized, type) else parameterized.__class__
        target = self.target if self.target is not None else param_class.__name__
        env_var = os.environ.get(self.varname, None)
        if env_var is None and self.json_file is None:
            return
        if self.json_file or env_var.endswith('.json'):
            try:
                fname = self.json_file if self.json_file else env_var
                with open(fullpath(fname), 'r') as f:
                    spec = json.load(f)
            except Exception:
                warnobj.warning('Could not load JSON file %r' % spec)
        else:
            spec = json.loads(env_var)
        if not isinstance(spec, dict):
            warnobj.warning('JSON parameter specification must be a dictionary.')
            return
        if target in spec:
            params = spec[target]
        else:
            params = spec
        for name, value in params.items():
            try:
                parameterized.param.update(**{name: value})
            except ValueError as e:
                warnobj.warning(str(e))