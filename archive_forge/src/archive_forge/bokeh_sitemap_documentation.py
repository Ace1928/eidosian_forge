from __future__ import annotations
import logging  # isort:skip
from html import escape
from os.path import join
from sphinx.errors import SphinxError
from sphinx.util.display import status_iterator
from . import PARALLEL_SAFE
 Required Sphinx extension setup function. 