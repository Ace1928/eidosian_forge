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
def transformed_data(self, row_limit: Optional[int]=None, exclude: Optional[Iterable[str]]=None) -> Optional[DataFrameLike]:
    """Evaluate a FacetChart's transforms

        Evaluate the data transforms associated with a FacetChart and return the
        transformed data a DataFrame

        Parameters
        ----------
        row_limit : int (optional)
            Maximum number of rows to return for each DataFrame. None (default) for unlimited
        exclude : iterable of str
            Set of the names of charts to exclude

        Returns
        -------
        DataFrame
            Transformed data as a DataFrame
        """
    from altair.utils._transformed_data import transformed_data
    return transformed_data(self, row_limit=row_limit, exclude=exclude)