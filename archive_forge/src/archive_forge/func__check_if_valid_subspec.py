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
def _check_if_valid_subspec(spec: Union[dict, core.SchemaBase], classname: str) -> None:
    """Check if the spec is a valid sub-spec.

    If it is not, then raise a ValueError
    """
    err = 'Objects with "{0}" attribute cannot be used within {1}. Consider defining the {0} attribute in the {1} object instead.'
    if not isinstance(spec, (core.SchemaBase, dict)):
        raise ValueError('Only chart objects can be used in {0}.'.format(classname))
    for attr in TOPLEVEL_ONLY_KEYS:
        if isinstance(spec, core.SchemaBase):
            val = getattr(spec, attr, Undefined)
        else:
            val = spec.get(attr, Undefined)
        if val is not Undefined:
            raise ValueError(err.format(attr, classname))