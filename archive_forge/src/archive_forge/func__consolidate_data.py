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
def _consolidate_data(data, context):
    """If data is specified inline, then move it to context['datasets']

    This function will modify context in-place, and return a new version of data
    """
    values = Undefined
    kwds = {}
    if isinstance(data, core.InlineData):
        if data.name is Undefined and data.values is not Undefined:
            if isinstance(data.values, core.InlineDataset):
                values = data.to_dict()['values']
            else:
                values = data.values
            kwds = {'format': data.format}
    elif isinstance(data, dict):
        if 'name' not in data and 'values' in data:
            values = data['values']
            kwds = {k: v for k, v in data.items() if k != 'values'}
    if values is not Undefined:
        name = _dataset_name(values)
        data = core.NamedData(name=name, **kwds)
        context.setdefault('datasets', {})[name] = values
    return data