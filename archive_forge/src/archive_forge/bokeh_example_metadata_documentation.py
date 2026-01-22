from __future__ import annotations
from sphinx.util import logging  # isort:skip
from docutils.parsers.rst.directives import unchanged
from sphinx.errors import SphinxError
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .templates import EXAMPLE_METADATA
from .util import get_sphinx_resources
 Required Sphinx extension setup function. 