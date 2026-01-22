import warnings
import hashlib
import io
import json
import jsonschema
import pandas as pd
from toolz.curried import pipe as _pipe
import itertools
import sys
from typing import cast, List, Optional, Any, Iterable, Union, Literal, IO
from typing import Type as TypingType
from typing import Dict as TypingDict
from .schema import core, channels, mixins, Undefined, UndefinedType, SCHEMA_URL
from .data import data_transformers
from ... import utils, expr
from ...expr import core as _expr_core
from .display import renderers, VEGALITE_VERSION, VEGAEMBED_VERSION, VEGA_VERSION
from .theme import themes
from .compiler import vegalite_compilers
from ...utils._vegafusion_data import (
from ...utils.core import DataFrameLike
from ...utils.data import DataType
def _repeat_names(params, repeat, spec):
    if params is Undefined:
        return params
    repeat = _get_repeat_strings(repeat)
    params_named = []
    for param in params:
        if not isinstance(param, core.TopLevelSelectionParameter):
            params_named.append(param)
            continue
        p = param.copy()
        views = []
        repeat_strings = _get_repeat_strings(repeat)
        for v in param.views:
            if isinstance(spec, Chart):
                if any((v.endswith(f'child__{r}') for r in repeat_strings)):
                    views.append(v)
                else:
                    views += [_extend_view_name(v, r, spec) for r in repeat_strings]
            elif isinstance(spec, LayerChart):
                if any((v.startswith(f'child__{r}') for r in repeat_strings)):
                    views.append(v)
                else:
                    views += [_extend_view_name(v, r, spec) for r in repeat_strings]
        p.views = views
        params_named.append(p)
    return params_named