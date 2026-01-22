from __future__ import annotations
import logging  # isort:skip
import importlib
import re
import textwrap
from os.path import basename
from sphinx.errors import SphinxError
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .templates import JINJA_DETAIL
 Required Sphinx extension setup function. 