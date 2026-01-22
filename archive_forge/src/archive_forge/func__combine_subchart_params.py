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
def _combine_subchart_params(params, subcharts):
    if params is Undefined:
        params = []
    param_info = []
    for param in params:
        p = _prepare_to_lift(param)
        param_info.append((p, _viewless_dict(p), [] if isinstance(p, core.VariableParameter) else p.views))
    subcharts = [subchart.copy() for subchart in subcharts]
    for subchart in subcharts:
        if not hasattr(subchart, 'params') or subchart.params is Undefined:
            continue
        if _needs_name(subchart):
            subchart.name = subchart._get_name()
        for param in subchart.params:
            p = _prepare_to_lift(param)
            pd = _viewless_dict(p)
            dlist = [d for _, d, _ in param_info]
            found = pd in dlist
            if isinstance(p, core.VariableParameter) and found:
                continue
            if isinstance(p, core.VariableParameter) and (not found):
                param_info.append((p, pd, []))
                continue
            if isinstance(subchart, Chart) and subchart.name not in p.views:
                p.views.append(subchart.name)
            if found:
                i = dlist.index(pd)
                _, _, old_views = param_info[i]
                new_views = [v for v in p.views if v not in old_views]
                old_views += new_views
            else:
                param_info.append((p, pd, p.views))
        subchart.params = Undefined
    for p, _, v in param_info:
        if len(v) > 0:
            p.views = v
    subparams = [p for p, _, _ in param_info]
    if len(subparams) == 0:
        subparams = Undefined
    return (subparams, subcharts)