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
def transform_timeunit(self, as_: Union[str, core.FieldName, UndefinedType]=Undefined, field: Union[str, core.FieldName, UndefinedType]=Undefined, timeUnit: Union[str, core.TimeUnit, UndefinedType]=Undefined, **kwargs: str) -> Self:
    """
        Add a :class:`TimeUnitTransform` to the schema.

        Parameters
        ----------
        as_ : string
            The output field to write the timeUnit value.
        field : string
            The data field to apply time unit.
        timeUnit : str or :class:`TimeUnit`
            The timeUnit.
        **kwargs
            transforms can also be passed by keyword argument; see Examples

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        Examples
        --------
        >>> import altair as alt
        >>> from altair import datum, expr

        >>> chart = alt.Chart().transform_timeunit(month='month(date)')
        >>> chart.transform[0]
        TimeUnitTransform({
          as: 'month',
          field: 'date',
          timeUnit: 'month'
        })

        It's also possible to pass the ``TimeUnitTransform`` arguments directly;
        this is most useful in cases where the desired field name is not a
        valid python identifier:

        >>> kwds = {'as': 'month', 'timeUnit': 'month', 'field': 'The Month'}
        >>> chart = alt.Chart().transform_timeunit(**kwds)
        >>> chart.transform[0]
        TimeUnitTransform({
          as: 'month',
          field: 'The Month',
          timeUnit: 'month'
        })

        As the first form is easier to write and understand, that is the
        recommended method.

        See Also
        --------
        alt.TimeUnitTransform : underlying transform object

        """
    if as_ is Undefined:
        as_ = kwargs.pop('as', Undefined)
    elif 'as' in kwargs:
        raise ValueError("transform_timeunit: both 'as_' and 'as' passed as arguments.")
    if as_ is not Undefined:
        dct = {'as': as_, 'timeUnit': timeUnit, 'field': field}
        self = self._add_transform(core.TimeUnitTransform(**dct))
    for as_, shorthand in kwargs.items():
        dct = utils.parse_shorthand(shorthand, parse_timeunits=True, parse_aggregates=False, parse_types=False)
        dct.pop('type', None)
        dct['as'] = as_
        if 'timeUnit' not in dct:
            raise ValueError("'{}' must include a valid timeUnit".format(shorthand))
        self = self._add_transform(core.TimeUnitTransform(**dct))
    return self