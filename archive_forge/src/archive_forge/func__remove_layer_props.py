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
def _remove_layer_props(chart, subcharts, layer_props):

    def remove_prop(subchart, prop):
        try:
            if subchart[prop] is not Undefined:
                subchart = subchart.copy()
                subchart[prop] = Undefined
        except KeyError:
            pass
        return subchart
    output_dict = {}
    if not subcharts:
        return (output_dict, subcharts)
    for prop in layer_props:
        if chart[prop] is Undefined:
            values = []
            for c in subcharts:
                try:
                    val = c[prop]
                    if val is not Undefined:
                        values.append(val)
                except KeyError:
                    pass
            if len(values) == 0:
                pass
            elif all((v == values[0] for v in values[1:])):
                output_dict[prop] = values[0]
            else:
                raise ValueError(f'There are inconsistent values {values} for {prop}')
        elif all((getattr(c, prop, Undefined) is Undefined or c[prop] == chart[prop] for c in subcharts)):
            output_dict[prop] = chart[prop]
        else:
            raise ValueError(f'There are inconsistent values {values} for {prop}')
        subcharts = [remove_prop(c, prop) for c in subcharts]
    return (output_dict, subcharts)