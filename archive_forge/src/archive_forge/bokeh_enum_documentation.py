from __future__ import annotations
import logging  # isort:skip
import importlib
import textwrap
from docutils.parsers.rst.directives import unchanged
from sphinx.errors import SphinxError
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .templates import ENUM_DETAIL
 Required Sphinx extension setup function. 