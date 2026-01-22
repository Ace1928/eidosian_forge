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
def _selection(type: Union[Literal['interval', 'point'], UndefinedType]=Undefined, **kwds) -> Parameter:
    param_kwds = {}
    for kwd in {'name', 'bind', 'value', 'empty', 'init', 'views'}:
        if kwd in kwds:
            param_kwds[kwd] = kwds.pop(kwd)
    select: Union[core.IntervalSelectionConfig, core.PointSelectionConfig]
    if type == 'interval':
        select = core.IntervalSelectionConfig(type=type, **kwds)
    elif type == 'point':
        select = core.PointSelectionConfig(type=type, **kwds)
    elif type in ['single', 'multi']:
        select = core.PointSelectionConfig(type='point', **kwds)
        warnings.warn('The types \'single\' and \'multi\' are now\n        combined and should be specified using "selection_point()".', utils.AltairDeprecationWarning, stacklevel=1)
    else:
        raise ValueError("'type' must be 'point' or 'interval'")
    return param(select=select, **param_kwds)