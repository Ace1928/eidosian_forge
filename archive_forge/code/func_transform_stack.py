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
def transform_stack(self, as_: Union[str, core.FieldName, List[str]], stack: Union[str, core.FieldName], groupby: List[Union[str, core.FieldName]], offset: Union[Literal['zero', 'center', 'normalize'], UndefinedType]=Undefined, sort: Union[List[core.SortField], UndefinedType]=Undefined) -> Self:
    """
        Add a :class:`StackTransform` to the schema.

        Parameters
        ----------
        as_ : anyOf(string, List(string))
            Output field names. This can be either a string or an array of strings with
            two elements denoting the name for the fields for stack start and stack end
            respectively.
            If a single string(eg."val") is provided, the end field will be "val_end".
        stack : string
            The field which is stacked.
        groupby : List(string)
            The data fields to group by.
        offset : enum('zero', 'center', 'normalize')
            Mode for stacking marks. Default: 'zero'.
        sort : List(:class:`SortField`)
            Field that determines the order of leaves in the stacked charts.

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        alt.StackTransform : underlying transform object
        """
    return self._add_transform(core.StackTransform(stack=stack, groupby=groupby, offset=offset, sort=sort, **{'as': as_}))