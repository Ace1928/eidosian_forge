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
def _combine_subchart_data(data, subcharts):

    def remove_data(subchart):
        if subchart.data is not Undefined:
            subchart = subchart.copy()
            subchart.data = Undefined
        return subchart
    if not subcharts:
        pass
    elif data is Undefined:
        subdata = subcharts[0].data
        if subdata is not Undefined and all((c.data is subdata for c in subcharts)):
            data = subdata
            subcharts = [remove_data(c) for c in subcharts]
    elif all((c.data is Undefined or c.data is data for c in subcharts)):
        subcharts = [remove_data(c) for c in subcharts]
    return (data, subcharts)