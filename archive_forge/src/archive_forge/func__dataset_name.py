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
def _dataset_name(values: Union[dict, list, core.InlineDataset]) -> str:
    """Generate a unique hash of the data

    Parameters
    ----------
    values : list, dict, core.InlineDataset
        A representation of data values.

    Returns
    -------
    name : string
        A unique name generated from the hash of the values.
    """
    if isinstance(values, core.InlineDataset):
        values = values.to_dict()
    if values == [{}]:
        return 'empty'
    values_json = json.dumps(values, sort_keys=True, default=str)
    hsh = hashlib.sha256(values_json.encode()).hexdigest()[:32]
    return 'data-' + hsh