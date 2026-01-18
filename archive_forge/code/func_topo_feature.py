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
def topo_feature(url: str, feature: str, **kwargs) -> core.UrlData:
    """A convenience function for extracting features from a topojson url

    Parameters
    ----------
    url : string
        An URL from which to load the data set.

    feature : string
        The name of the TopoJSON object set to convert to a GeoJSON feature collection. For
        example, in a map of the world, there may be an object set named `"countries"`.
        Using the feature property, we can extract this set and generate a GeoJSON feature
        object for each country.

    **kwargs :
        additional keywords passed to TopoDataFormat
    """
    return core.UrlData(url=url, format=core.TopoDataFormat(type='topojson', feature=feature, **kwargs))