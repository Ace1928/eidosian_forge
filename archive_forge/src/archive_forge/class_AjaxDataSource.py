from __future__ import annotations
import logging # isort:skip
from typing import (
import numpy as np
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from ..util.deprecation import deprecated
from ..util.serialization import convert_datetime_array
from ..util.warnings import BokehUserWarning, warn
from .callbacks import CustomJS
from .filters import AllIndices, Filter, IntersectionFilter
from .selections import Selection, SelectionPolicy, UnionRenderers
class AjaxDataSource(WebDataSource):
    """ A data source that can populate columns by making Ajax calls to REST
    endpoints.

    The ``AjaxDataSource`` can be especially useful if you want to make a
    standalone document (i.e. not backed by the Bokeh server) that can still
    dynamically update using an existing REST API.

    The response from the REST API should match the ``.data`` property of a
    standard ``ColumnDataSource``, i.e. a JSON dict that maps names to arrays
    of values:

    .. code-block:: python

        {
            'x' : [1, 2, 3, ...],
            'y' : [9, 3, 2, ...]
        }

    Alternatively, if the REST API returns a different format, a ``CustomJS``
    callback can be provided to convert the REST response into Bokeh format,
    via the ``adapter`` property of this data source.

    Initial data can be set by specifying the ``data`` property directly.
    This is necessary when used in conjunction with a ``FactorRange``, even
    if the columns in `data`` are empty.

    A full example can be seen at :bokeh-tree:`examples/basic/data/ajax_source.py`

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    polling_interval = Nullable(Int, help='\n    A polling interval (in milliseconds) for updating data source.\n    ')
    method = Enum('POST', 'GET', help='\n    Specify the HTTP method to use for the Ajax request (GET or POST)\n    ')
    if_modified = Bool(False, help='\n    Whether to include an ``If-Modified-Since`` header in Ajax requests\n    to the server. If this header is supported by the server, then only\n    new data since the last request will be returned.\n    ')
    content_type = String(default='application/json', help='\n    Set the "contentType" parameter for the Ajax request.\n    ')
    http_headers = Dict(String, String, help="\n    Specify HTTP headers to set for the Ajax request.\n\n    Example:\n\n    .. code-block:: python\n\n        ajax_source.headers = { 'x-my-custom-header': 'some value' }\n\n    ")