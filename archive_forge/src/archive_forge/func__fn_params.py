from a list that contains the shared attribute. This is required for the
import abc
import operator
import sys
from functools import partial
from packaging.version import Version
from types import FunctionType, MethodType
import holoviews as hv
import pandas as pd
import panel as pn
import param
from panel.layout import Column, Row, VSpacer, HSpacer
from panel.util import get_method_owner, full_groupby
from panel.widgets.base import Widget
from .converter import HoloViewsConverter
from .util import (
@property
def _fn_params(self):
    if self._fn is None:
        deps = []
    elif isinstance(self._fn, pn.param.ParamFunction):
        dinfo = getattr(self._fn.object, '_dinfo', {})
        deps = list(dinfo.get('dependencies', [])) + list(dinfo.get('kw', {}).values())
    else:
        parameterized = get_method_owner(self._fn.object)
        deps = parameterized.param.method_dependencies(self._fn.object.__name__)
    return deps