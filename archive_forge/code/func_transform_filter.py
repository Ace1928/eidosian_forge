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
def transform_filter(self, filter: Union[str, core.Expr, _expr_core.Expression, core.Predicate, Parameter, core.PredicateComposition, TypingDict[str, Union[core.Predicate, str, list, bool]]], **kwargs) -> Self:
    """
        Add a :class:`FilterTransform` to the schema.

        Parameters
        ----------
        filter : a filter expression or :class:`PredicateComposition`
            The `filter` property must be one of the predicate definitions:
            (1) a string or alt.expr expression
            (2) a range predicate
            (3) a selection predicate
            (4) a logical operand combining (1)-(3)
            (5) a Selection object

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining
        """
    if isinstance(filter, Parameter):
        new_filter: TypingDict[str, Union[bool, str]] = {'param': filter.name}
        if 'empty' in kwargs:
            new_filter['empty'] = kwargs.pop('empty')
        elif isinstance(filter.empty, bool):
            new_filter['empty'] = filter.empty
        filter = new_filter
    return self._add_transform(core.FilterTransform(filter=filter, **kwargs))