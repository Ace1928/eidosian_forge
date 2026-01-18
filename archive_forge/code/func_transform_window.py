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
def transform_window(self, window: Union[List[core.WindowFieldDef], UndefinedType]=Undefined, frame: Union[List[Optional[int]], UndefinedType]=Undefined, groupby: Union[List[str], UndefinedType]=Undefined, ignorePeers: Union[bool, UndefinedType]=Undefined, sort: Union[List[Union[core.SortField, TypingDict[str, str]]], UndefinedType]=Undefined, **kwargs: str) -> Self:
    """Add a :class:`WindowTransform` to the schema

        Parameters
        ----------
        window : List(:class:`WindowFieldDef`)
            The definition of the fields in the window, and what calculations to use.
        frame : List(anyOf(None, int))
            A frame specification as a two-element array indicating how the sliding window
            should proceed. The array entries should either be a number indicating the offset
            from the current data object, or null to indicate unbounded rows preceding or
            following the current data object. The default value is ``[null, 0]``, indicating
            that the sliding window includes the current object and all preceding objects. The
            value ``[-5, 5]`` indicates that the window should include five objects preceding
            and five objects following the current object. Finally, ``[null, null]`` indicates
            that the window frame should always include all data objects. The only operators
            affected are the aggregation operations and the ``first_value``, ``last_value``, and
            ``nth_value`` window operations. The other window operations are not affected by
            this.

            **Default value:** :  ``[null, 0]`` (includes the current object and all preceding
            objects)
        groupby : List(string)
            The data fields for partitioning the data objects into separate windows. If
            unspecified, all data points will be in a single group.
        ignorePeers : boolean
            Indicates if the sliding window frame should ignore peer values. (Peer values are
            those considered identical by the sort criteria). The default is false, causing the
            window frame to expand to include all peer values. If set to true, the window frame
            will be defined by offset values only. This setting only affects those operations
            that depend on the window frame, namely aggregation operations and the first_value,
            last_value, and nth_value window operations.

            **Default value:** ``false``
        sort : List(:class:`SortField`)
            A sort field definition for sorting data objects within a window. If two data
            objects are considered equal by the comparator, they are considered “peer” values of
            equal rank. If sort is not specified, the order is undefined: data objects are
            processed in the order they are observed and none are considered peers (the
            ignorePeers parameter is ignored and treated as if set to ``true`` ).
        **kwargs
            transforms can also be passed by keyword argument; see Examples

        Examples
        --------
        A cumulative line chart

        >>> import altair as alt
        >>> import numpy as np
        >>> import pandas as pd
        >>> data = pd.DataFrame({'x': np.arange(100),
        ...                      'y': np.random.randn(100)})
        >>> chart = alt.Chart(data).mark_line().encode(
        ...     x='x:Q',
        ...     y='ycuml:Q'
        ... ).transform_window(
        ...     ycuml='sum(y)'
        ... )
        >>> chart.transform[0]
        WindowTransform({
          window: [WindowFieldDef({
            as: 'ycuml',
            field: 'y',
            op: 'sum'
          })]
        })

        """
    if kwargs:
        if window is Undefined:
            window = []
        for as_, shorthand in kwargs.items():
            kwds = {'as': as_}
            kwds.update(utils.parse_shorthand(shorthand, parse_aggregates=False, parse_window_ops=True, parse_timeunits=False, parse_types=False))
            assert not isinstance(window, UndefinedType)
            window.append(core.WindowFieldDef(**kwds))
    return self._add_transform(core.WindowTransform(window=window, frame=frame, groupby=groupby, ignorePeers=ignorePeers, sort=sort))