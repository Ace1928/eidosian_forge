from __future__ import annotations
import logging  # isort:skip
import importlib
import json
import warnings
from os import getenv
from typing import Any
from docutils.parsers.rst.directives import unchanged
from sphinx.errors import SphinxError
from bokeh.core.property.singletons import Undefined
from bokeh.core.serialization import AnyRep, Serializer, SymbolRep
from bokeh.model import Model
from bokeh.util.warnings import BokehDeprecationWarning
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective, py_sig_re
from .templates import MODEL_DETAIL
class DocsSerializer(Serializer):

    def _encode(self, obj: Any) -> AnyRep:
        if obj is Undefined:
            return SymbolRep(type='symbol', name='unset')
        else:
            return super()._encode(obj)