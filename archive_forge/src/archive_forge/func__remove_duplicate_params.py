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
def _remove_duplicate_params(layer):
    subcharts = [subchart.copy() for subchart in layer]
    found_params = []
    for subchart in subcharts:
        if not hasattr(subchart, 'params') or subchart.params is Undefined:
            continue
        params = []
        for param in subchart.params:
            if isinstance(param, core.VariableParameter):
                params.append(param)
                continue
            p = param.copy()
            pd = _viewless_dict(p)
            if pd not in found_params:
                params.append(p)
                found_params.append(pd)
        if len(params) == 0:
            subchart.params = Undefined
        else:
            subchart.params = params
    return subcharts